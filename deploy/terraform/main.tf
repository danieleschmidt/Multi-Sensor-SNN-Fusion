# Multi-Sensor SNN-Fusion Global Infrastructure
# Terraform configuration for multi-region deployment with GDPR/CCPA compliance

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
  
  backend "s3" {
    bucket  = "terragon-snn-fusion-terraform-state"
    key     = "global/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
    
    dynamodb_table = "terragon-terraform-locks"
  }
}

# Local variables
locals {
  project_name = "snn-fusion"
  environment  = var.environment
  
  # Global regions for deployment
  regions = {
    "us-east-1"      = { name = "North America East", timezone = "America/New_York" }
    "us-west-2"      = { name = "North America West", timezone = "America/Los_Angeles" }
    "eu-west-1"      = { name = "Europe West", timezone = "Europe/London" }
    "eu-central-1"   = { name = "Europe Central", timezone = "Europe/Berlin" }
    "ap-southeast-1" = { name = "Asia Pacific Southeast", timezone = "Asia/Singapore" }
    "ap-northeast-1" = { name = "Asia Pacific Northeast", timezone = "Asia/Tokyo" }
  }
  
  # GDPR/CCPA compliance regions
  gdpr_regions = ["eu-west-1", "eu-central-1"]
  ccpa_regions = ["us-west-2"]
  
  common_tags = {
    Project     = local.project_name
    Environment = local.environment
    ManagedBy   = "terraform"
    Owner       = "terragon-labs"
    Compliance  = "gdpr-ccpa"
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "enable_gpu" {
  description = "Enable GPU instances for neuromorphic computing"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable comprehensive monitoring"
  type        = bool
  default     = true
}

variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Global resources
module "global_vpc" {
  source = "./modules/vpc"
  
  for_each = local.regions
  
  providers = {
    aws = aws
  }
  
  region             = each.key
  environment        = local.environment
  availability_zones = data.aws_availability_zones.available.names
  
  # VPC Configuration
  vpc_cidr = "10.${index(keys(local.regions), each.key)}.0.0/16"
  
  # Subnets
  public_subnet_cidrs  = ["10.${index(keys(local.regions), each.key)}.1.0/24", "10.${index(keys(local.regions), each.key)}.2.0/24"]
  private_subnet_cidrs = ["10.${index(keys(local.regions), each.key)}.10.0/24", "10.${index(keys(local.regions), each.key)}.20.0/24"]
  
  # NAT Gateway for private subnets
  enable_nat_gateway = true
  single_nat_gateway = false
  
  # DNS
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(local.common_tags, {
    Region = each.value.name
  })
}

# EKS Clusters for each region
module "eks_cluster" {
  source = "./modules/eks"
  
  for_each = local.regions
  
  providers = {
    aws        = aws
    kubernetes = kubernetes
  }
  
  cluster_name    = "${local.project_name}-${each.key}"
  cluster_version = "1.28"
  
  vpc_id         = module.global_vpc[each.key].vpc_id
  subnet_ids     = module.global_vpc[each.key].private_subnet_ids
  
  # Node groups
  node_groups = {
    general = {
      instance_types = ["t3.large", "t3.xlarge"]
      scaling_config = {
        desired_size = 2
        max_size     = 10
        min_size     = 1
      }
      disk_size = 50
    }
    
    compute_optimized = {
      instance_types = ["c5.2xlarge", "c5.4xlarge"]
      scaling_config = {
        desired_size = 1
        max_size     = 5
        min_size     = 0
      }
      disk_size = 100
      
      taints = [{
        key    = "compute-optimized"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
    
    # GPU nodes for neuromorphic computing
    gpu = var.enable_gpu ? {
      instance_types = ["p3.2xlarge", "g4dn.xlarge"]
      ami_type       = "AL2_x86_64_GPU"
      scaling_config = {
        desired_size = 0
        max_size     = 3
        min_size     = 0
      }
      disk_size = 200
      
      taints = [{
        key    = "gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    } : {}
  }
  
  # RBAC
  manage_aws_auth_configmap = true
  aws_auth_users = [
    {
      userarn  = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:user/snn-fusion-admin"
      username = "snn-fusion-admin"
      groups   = ["system:masters"]
    }
  ]
  
  # Security
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  cluster_endpoint_public_access_cidrs = ["0.0.0.0/0"]
  
  # Encryption
  cluster_encryption_config = [{
    provider_key_arn = aws_kms_key.eks[each.key].arn
    resources        = ["secrets"]
  }]
  
  # Logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  tags = merge(local.common_tags, {
    Region = each.value.name
  })
}

# KMS Keys for encryption
resource "aws_kms_key" "eks" {
  for_each = local.regions
  
  description             = "EKS encryption key for ${each.key}"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Name   = "${local.project_name}-eks-${each.key}"
    Region = each.value.name
  })
}

resource "aws_kms_alias" "eks" {
  for_each = local.regions
  
  name          = "alias/${local.project_name}-eks-${each.key}"
  target_key_id = aws_kms_key.eks[each.key].key_id
}

# RDS for data persistence with global replication
module "rds_global" {
  source = "./modules/rds"
  
  # Primary region
  primary_region = "us-east-1"
  
  # Read replicas in other regions
  read_replica_regions = [for region in keys(local.regions) : region if region != "us-east-1"]
  
  # Database configuration
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r6g.large"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.rds.arn
  
  # High availability
  multi_az = true
  
  # Backup configuration
  backup_retention_period = var.enable_backup ? 30 : 0
  backup_window          = "03:00-04:00"
  maintenance_window     = "Sun:04:00-Sun:05:00"
  
  # Monitoring
  monitoring_interval = var.enable_monitoring ? 60 : 0
  monitoring_role_arn = var.enable_monitoring ? aws_iam_role.rds_monitoring.arn : null
  
  # Security
  vpc_security_group_ids = [aws_security_group.rds.id]
  subnet_group_name      = aws_db_subnet_group.main.name
  
  # GDPR/CCPA compliance
  deletion_protection = contains(local.gdpr_regions, "us-east-1") || contains(local.ccpa_regions, "us-east-1")
  
  tags = local.common_tags
}

# ElastiCache for caching
resource "aws_elasticache_replication_group" "main" {
  for_each = local.regions
  
  replication_group_id         = "${local.project_name}-cache-${each.key}"
  description                  = "SNN-Fusion cache cluster for ${each.value.name}"
  
  node_type                    = "cache.r6g.large"
  num_cache_clusters           = 2
  port                         = 6379
  parameter_group_name         = "default.redis7"
  
  # Security
  subnet_group_name            = aws_elasticache_subnet_group.main[each.key].name
  security_group_ids           = [aws_security_group.elasticache[each.key].id]
  at_rest_encryption_enabled   = true
  transit_encryption_enabled   = true
  auth_token                   = random_password.redis_auth[each.key].result
  
  # Backup
  snapshot_retention_limit     = var.enable_backup ? 7 : 0
  snapshot_window             = "03:00-05:00"
  
  # Maintenance
  maintenance_window          = "sun:05:00-sun:07:00"
  
  # Global datastore for cross-region replication
  global_replication_group_id = each.key == "us-east-1" ? aws_elasticache_global_replication_group.main.global_replication_group_id : null
  
  tags = merge(local.common_tags, {
    Region = each.value.name
  })
}

# Global replication for Redis
resource "aws_elasticache_global_replication_group" "main" {
  global_replication_group_id_suffix = "${local.project_name}-global"
  description                        = "Global replication group for SNN-Fusion"
  
  primary_replication_group_id = aws_elasticache_replication_group.main["us-east-1"].replication_group_id
}

# S3 buckets for data storage with cross-region replication
module "s3_global" {
  source = "./modules/s3"
  
  for_each = local.regions
  
  bucket_name = "${local.project_name}-data-${each.key}"
  
  # Versioning and lifecycle
  versioning_enabled = true
  lifecycle_rules = [
    {
      id                                     = "intelligent_tiering"
      status                                = "Enabled"
      intelligent_tiering_configurations = [{
        id     = "entire_bucket"
        status = "Enabled"
      }]
    },
    {
      id     = "delete_old_versions"
      status = "Enabled"
      noncurrent_version_expiration = {
        noncurrent_days = 90
      }
    }
  ]
  
  # Cross-region replication
  replication_configuration = each.key == "us-east-1" ? {
    role = aws_iam_role.s3_replication.arn
    rules = [
      for region in keys(local.regions) : {
        id       = "replicate-to-${region}"
        status   = "Enabled"
        priority = index(keys(local.regions), region)
        
        destination = {
          bucket             = "arn:aws:s3:::${local.project_name}-data-${region}"
          storage_class      = "STANDARD_IA"
          replica_kms_key_id = aws_kms_key.s3[region].arn
        }
      } if region != "us-east-1"
    ]
  } : null
  
  # Encryption
  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        kms_master_key_id = aws_kms_key.s3[each.key].arn
        sse_algorithm     = "aws:kms"
      }
      bucket_key_enabled = true
    }
  }
  
  # Public access block (security)
  public_access_block = {
    block_public_acls       = true
    block_public_policy     = true
    ignore_public_acls      = true
    restrict_public_buckets = true
  }
  
  # GDPR/CCPA compliance
  object_lock_enabled = contains(local.gdpr_regions, each.key) || contains(local.ccpa_regions, each.key)
  
  tags = merge(local.common_tags, {
    Region     = each.value.name
    DataClass  = "neuromorphic-data"
    Compliance = contains(local.gdpr_regions, each.key) ? "gdpr" : contains(local.ccpa_regions, each.key) ? "ccpa" : "standard"
  })
}

# CloudWatch for monitoring
module "monitoring" {
  source = "./modules/monitoring"
  
  count = var.enable_monitoring ? 1 : 0
  
  project_name = local.project_name
  environment  = local.environment
  
  # Metrics
  custom_metrics = [
    "snn_fusion_requests_per_second",
    "snn_fusion_model_inference_time",
    "snn_fusion_memory_usage",
    "snn_fusion_gpu_utilization"
  ]
  
  # Alarms
  alarms = [
    {
      name                = "high-cpu-usage"
      metric_name         = "CPUUtilization"
      threshold           = 80
      comparison_operator = "GreaterThanThreshold"
      evaluation_periods  = 2
    },
    {
      name                = "high-memory-usage"
      metric_name         = "MemoryUtilization"
      threshold           = 85
      comparison_operator = "GreaterThanThreshold"
      evaluation_periods  = 2
    },
    {
      name                = "high-error-rate"
      metric_name         = "ErrorRate"
      threshold           = 5
      comparison_operator = "GreaterThanThreshold"
      evaluation_periods  = 3
    }
  ]
  
  # SNS topics for alerts
  notification_endpoints = [
    "alerts@terragonlabs.com",
    "snn-fusion-team@terragonlabs.com"
  ]
  
  tags = local.common_tags
}

# WAF for security
resource "aws_wafv2_web_acl" "main" {
  name  = "${local.project_name}-waf"
  scope = "REGIONAL"
  
  default_action {
    allow {}
  }
  
  # Rate limiting
  rule {
    name     = "RateLimitRule"
    priority = 1
    
    override_action {
      none {}
    }
    
    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }
    
    action {
      block {}
    }
  }
  
  # AWS Managed Rules
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 10
    
    override_action {
      none {}
    }
    
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }
  
  # IP reputation
  rule {
    name     = "AWSManagedRulesAmazonIpReputationList"
    priority = 20
    
    override_action {
      none {}
    }
    
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesAmazonIpReputationList"
        vendor_name = "AWS"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "IpReputationListMetric"
      sampled_requests_enabled   = true
    }
  }
  
  tags = local.common_tags
  
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${local.project_name}WebAcl"
    sampled_requests_enabled   = true
  }
}

# Route53 for global DNS
resource "aws_route53_zone" "main" {
  name = "snn-fusion.terragonlabs.com"
  
  tags = local.common_tags
}

# Health checks for each region
resource "aws_route53_health_check" "regional" {
  for_each = local.regions
  
  fqdn                            = "${each.key}.snn-fusion.terragonlabs.com"
  port                            = 443
  type                            = "HTTPS"
  resource_path                   = "/health"
  failure_threshold               = 3
  request_interval                = 30
  
  tags = merge(local.common_tags, {
    Region = each.value.name
  })
}

# Weighted routing for global load balancing
resource "aws_route53_record" "global" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "api.snn-fusion.terragonlabs.com"
  type    = "A"
  
  for_each = local.regions
  
  set_identifier = each.key
  
  weighted_routing_policy {
    weight = 100  # Equal weight for all regions
  }
  
  health_check_id = aws_route53_health_check.regional[each.key].id
  
  alias {
    name                   = module.eks_cluster[each.key].cluster_endpoint
    zone_id               = module.eks_cluster[each.key].cluster_hosted_zone_id
    evaluate_target_health = true
  }
}

# Outputs
output "cluster_endpoints" {
  description = "EKS cluster endpoints for each region"
  value = {
    for region, cluster in module.eks_cluster : region => cluster.cluster_endpoint
  }
}

output "s3_buckets" {
  description = "S3 bucket names for each region"
  value = {
    for region, bucket in module.s3_global : region => bucket.bucket_name
  }
}

output "dns_zone" {
  description = "Route53 hosted zone"
  value = {
    zone_id     = aws_route53_zone.main.zone_id
    name_servers = aws_route53_zone.main.name_servers
  }
}

output "global_endpoint" {
  description = "Global API endpoint"
  value       = "https://api.snn-fusion.terragonlabs.com"
}