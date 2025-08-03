"""
Notification Services for SNN-Fusion

Implements email and Slack notification services for neuromorphic
computing workflows and experiment updates.
"""

import os
import json
import smtplib
import requests
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
import jinja2


class EmailNotificationService:
    """
    Email notification service for experiment updates and alerts.
    
    Supports HTML templates, attachments, and SMTP configuration
    for sending neuromorphic experiment notifications.
    """
    
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        template_dir: Optional[str] = None,
    ):
        """
        Initialize email service.
        
        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            use_tls: Whether to use TLS encryption
            template_dir: Directory containing email templates
        """
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port
        self.username = username or os.getenv('SMTP_USERNAME')
        self.password = password or os.getenv('SMTP_PASSWORD')
        self.use_tls = use_tls
        self.logger = logging.getLogger(__name__)
        
        # Setup template environment
        template_path = template_dir or os.path.join(
            os.path.dirname(__file__), '..', 'templates', 'email'
        )
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_path),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        if not self.username or not self.password:
            self.logger.warning("Email credentials not configured")
    
    def send_experiment_notification(
        self,
        recipients: List[str],
        experiment_data: Dict[str, Any],
        notification_type: str = 'update',
    ) -> bool:
        """
        Send experiment notification email.
        
        Args:
            recipients: List of recipient email addresses
            experiment_data: Experiment information
            notification_type: Type of notification (created, started, completed, failed)
            
        Returns:
            Success status
        """
        try:
            # Prepare email content
            subject = self._generate_experiment_subject(experiment_data, notification_type)
            html_content = self._render_experiment_template(experiment_data, notification_type)
            
            return self.send_email(
                recipients=recipients,
                subject=subject,
                html_content=html_content,
                metadata={
                    'experiment_id': experiment_data.get('id'),
                    'notification_type': notification_type
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send experiment notification: {e}")
            return False
    
    def send_training_alert(
        self,
        recipients: List[str],
        training_data: Dict[str, Any],
        alert_type: str = 'completion',
    ) -> bool:
        """
        Send training completion or failure alert.
        
        Args:
            recipients: List of recipient email addresses
            training_data: Training run information
            alert_type: Type of alert (completion, failure, milestone)
            
        Returns:
            Success status
        """
        try:
            subject = self._generate_training_subject(training_data, alert_type)
            html_content = self._render_training_template(training_data, alert_type)
            
            # Attach training logs if available
            attachments = []
            if training_data.get('log_file') and os.path.exists(training_data['log_file']):
                attachments.append(training_data['log_file'])
            
            return self.send_email(
                recipients=recipients,
                subject=subject,
                html_content=html_content,
                attachments=attachments,
                metadata={
                    'training_id': training_data.get('id'),
                    'alert_type': alert_type
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send training alert: {e}")
            return False
    
    def send_hardware_deployment_notification(
        self,
        recipients: List[str],
        deployment_data: Dict[str, Any],
    ) -> bool:
        """
        Send hardware deployment notification.
        
        Args:
            recipients: List of recipient email addresses
            deployment_data: Deployment information
            
        Returns:
            Success status
        """
        try:
            subject = f"Hardware Deployment - {deployment_data.get('hardware_type', 'Unknown')} - {deployment_data.get('status', 'Status')}"
            html_content = self._render_deployment_template(deployment_data)
            
            return self.send_email(
                recipients=recipients,
                subject=subject,
                html_content=html_content,
                metadata={
                    'deployment_id': deployment_data.get('id'),
                    'hardware_type': deployment_data.get('hardware_type')
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send deployment notification: {e}")
            return False
    
    def send_email(
        self,
        recipients: List[str],
        subject: str,
        text_content: Optional[str] = None,
        html_content: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send email with optional attachments.
        
        Args:
            recipients: List of recipient email addresses
            subject: Email subject
            text_content: Plain text content
            html_content: HTML content
            attachments: List of file paths to attach
            metadata: Additional metadata for logging
            
        Returns:
            Success status
        """
        if not self.username or not self.password:
            self.logger.error("Email credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Add text content
            if text_content:
                text_part = MIMEText(text_content, 'plain')
                msg.attach(text_part)
            
            # Add HTML content
            if html_content:
                html_part = MIMEText(html_content, 'html')
                msg.attach(html_part)
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                        
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {os.path.basename(file_path)}'
                        )
                        msg.attach(part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.logger.info(f"Email sent successfully to {recipients}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def _generate_experiment_subject(self, experiment_data: Dict[str, Any], notification_type: str) -> str:
        """Generate experiment notification subject."""
        experiment_name = experiment_data.get('name', 'Unknown Experiment')
        type_map = {
            'created': 'Created',
            'started': 'Started',
            'completed': 'Completed',
            'failed': 'Failed',
            'update': 'Updated'
        }
        status = type_map.get(notification_type, 'Update')
        return f"SNN-Fusion Experiment {status}: {experiment_name}"
    
    def _generate_training_subject(self, training_data: Dict[str, Any], alert_type: str) -> str:
        """Generate training alert subject."""
        model_name = training_data.get('model_name', 'Unknown Model')
        type_map = {
            'completion': 'Training Completed',
            'failure': 'Training Failed',
            'milestone': 'Training Milestone',
            'started': 'Training Started'
        }
        status = type_map.get(alert_type, 'Training Update')
        return f"SNN-Fusion {status}: {model_name}"
    
    def _render_experiment_template(self, experiment_data: Dict[str, Any], notification_type: str) -> str:
        """Render experiment notification template."""
        try:
            template = self.template_env.get_template('experiment_notification.html')
            return template.render(
                experiment=experiment_data,
                notification_type=notification_type,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            self.logger.warning(f"Failed to render template: {e}")
            return self._generate_fallback_html(experiment_data, notification_type)
    
    def _render_training_template(self, training_data: Dict[str, Any], alert_type: str) -> str:
        """Render training alert template."""
        try:
            template = self.template_env.get_template('training_alert.html')
            return template.render(
                training=training_data,
                alert_type=alert_type,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            self.logger.warning(f"Failed to render template: {e}")
            return self._generate_fallback_training_html(training_data, alert_type)
    
    def _render_deployment_template(self, deployment_data: Dict[str, Any]) -> str:
        """Render deployment notification template."""
        try:
            template = self.template_env.get_template('deployment_notification.html')
            return template.render(
                deployment=deployment_data,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            self.logger.warning(f"Failed to render template: {e}")
            return self._generate_fallback_deployment_html(deployment_data)
    
    def _generate_fallback_html(self, experiment_data: Dict[str, Any], notification_type: str) -> str:
        """Generate fallback HTML content."""
        return f"""
        <html>
        <body>
            <h2>SNN-Fusion Experiment {notification_type.title()}</h2>
            <p><strong>Experiment:</strong> {experiment_data.get('name', 'Unknown')}</p>
            <p><strong>Description:</strong> {experiment_data.get('description', 'No description')}</p>
            <p><strong>Status:</strong> {experiment_data.get('status', 'Unknown')}</p>
            <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """
    
    def _generate_fallback_training_html(self, training_data: Dict[str, Any], alert_type: str) -> str:
        """Generate fallback training HTML content."""
        return f"""
        <html>
        <body>
            <h2>SNN-Fusion Training {alert_type.title()}</h2>
            <p><strong>Model:</strong> {training_data.get('model_name', 'Unknown')}</p>
            <p><strong>Status:</strong> {training_data.get('status', 'Unknown')}</p>
            <p><strong>Accuracy:</strong> {training_data.get('accuracy', 'N/A')}</p>
            <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """
    
    def _generate_fallback_deployment_html(self, deployment_data: Dict[str, Any]) -> str:
        """Generate fallback deployment HTML content."""
        return f"""
        <html>
        <body>
            <h2>SNN-Fusion Hardware Deployment</h2>
            <p><strong>Hardware:</strong> {deployment_data.get('hardware_type', 'Unknown')}</p>
            <p><strong>Model:</strong> {deployment_data.get('model_name', 'Unknown')}</p>
            <p><strong>Status:</strong> {deployment_data.get('status', 'Unknown')}</p>
            <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """


class SlackNotificationService:
    """
    Slack notification service for real-time experiment updates.
    
    Integrates with Slack webhooks and bot API for sending
    formatted notifications about neuromorphic experiments.
    """
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        bot_token: Optional[str] = None,
        default_channel: str = '#experiments',
    ):
        """
        Initialize Slack service.
        
        Args:
            webhook_url: Slack webhook URL
            bot_token: Slack bot token
            default_channel: Default channel for notifications
        """
        self.webhook_url = webhook_url or os.getenv('SLACK_WEBHOOK_URL')
        self.bot_token = bot_token or os.getenv('SLACK_BOT_TOKEN')
        self.default_channel = default_channel
        self.logger = logging.getLogger(__name__)
        
        if not self.webhook_url and not self.bot_token:
            self.logger.warning("No Slack credentials configured")
    
    def send_experiment_update(
        self,
        experiment_data: Dict[str, Any],
        update_type: str = 'update',
        channel: Optional[str] = None,
    ) -> bool:
        """
        Send experiment update to Slack.
        
        Args:
            experiment_data: Experiment information
            update_type: Type of update (created, started, completed, failed)
            channel: Slack channel (optional)
            
        Returns:
            Success status
        """
        try:
            message = self._format_experiment_message(experiment_data, update_type)
            return self.send_message(
                message=message,
                channel=channel or self.default_channel
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send experiment update: {e}")
            return False
    
    def send_training_notification(
        self,
        training_data: Dict[str, Any],
        notification_type: str = 'completion',
        channel: Optional[str] = None,
    ) -> bool:
        """
        Send training notification to Slack.
        
        Args:
            training_data: Training run information
            notification_type: Type of notification
            channel: Slack channel (optional)
            
        Returns:
            Success status
        """
        try:
            message = self._format_training_message(training_data, notification_type)
            return self.send_message(
                message=message,
                channel=channel or self.default_channel
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send training notification: {e}")
            return False
    
    def send_alert(
        self,
        alert_data: Dict[str, Any],
        severity: str = 'info',
        channel: Optional[str] = None,
    ) -> bool:
        """
        Send alert notification to Slack.
        
        Args:
            alert_data: Alert information
            severity: Alert severity (info, warning, error, critical)
            channel: Slack channel (optional)
            
        Returns:
            Success status
        """
        try:
            message = self._format_alert_message(alert_data, severity)
            return self.send_message(
                message=message,
                channel=channel or self.default_channel
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
            return False
    
    def send_message(
        self,
        message: Dict[str, Any],
        channel: Optional[str] = None,
    ) -> bool:
        """
        Send message to Slack using webhook or bot API.
        
        Args:
            message: Slack message payload
            channel: Target channel
            
        Returns:
            Success status
        """
        if self.webhook_url:
            return self._send_webhook_message(message)
        elif self.bot_token:
            return self._send_bot_message(message, channel)
        else:
            self.logger.error("No Slack credentials configured")
            return False
    
    def _send_webhook_message(self, message: Dict[str, Any]) -> bool:
        """Send message via webhook."""
        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            
            self.logger.info("Slack webhook message sent successfully")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to send webhook message: {e}")
            return False
    
    def _send_bot_message(self, message: Dict[str, Any], channel: Optional[str]) -> bool:
        """Send message via bot API."""
        try:
            url = "https://slack.com/api/chat.postMessage"
            headers = {
                'Authorization': f'Bearer {self.bot_token}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'channel': channel or self.default_channel,
                **message
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if result.get('ok'):
                self.logger.info("Slack bot message sent successfully")
                return True
            else:
                self.logger.error(f"Slack API error: {result.get('error')}")
                return False
                
        except requests.RequestException as e:
            self.logger.error(f"Failed to send bot message: {e}")
            return False
    
    def _format_experiment_message(self, experiment_data: Dict[str, Any], update_type: str) -> Dict[str, Any]:
        """Format experiment update message."""
        experiment_name = experiment_data.get('name', 'Unknown Experiment')
        status = experiment_data.get('status', 'unknown')
        
        # Choose emoji based on update type
        emoji_map = {
            'created': ':test_tube:',
            'started': ':rocket:',
            'completed': ':white_check_mark:',
            'failed': ':x:',
            'update': ':information_source:'
        }
        emoji = emoji_map.get(update_type, ':gear:')
        
        # Choose color based on status
        color_map = {
            'created': '#36a64f',    # green
            'started': '#ffaa00',    # orange
            'completed': '#36a64f',  # green
            'failed': '#ff0000',     # red
            'update': '#3AA3E3'      # blue
        }
        color = color_map.get(update_type, '#808080')
        
        # Build message
        text = f"{emoji} Experiment {update_type.title()}: *{experiment_name}*"
        
        fields = [
            {
                "title": "Status",
                "value": status,
                "short": True
            },
            {
                "title": "Configuration",
                "value": f"Model: {experiment_data.get('config', {}).get('model_type', 'Unknown')}",
                "short": True
            }
        ]
        
        if experiment_data.get('description'):
            fields.append({
                "title": "Description",
                "value": experiment_data['description'],
                "short": False
            })
        
        return {
            "text": text,
            "attachments": [
                {
                    "color": color,
                    "fields": fields,
                    "footer": "SNN-Fusion",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }
    
    def _format_training_message(self, training_data: Dict[str, Any], notification_type: str) -> Dict[str, Any]:
        """Format training notification message."""
        model_name = training_data.get('model_name', 'Unknown Model')
        
        emoji_map = {
            'started': ':hourglass_flowing_sand:',
            'completion': ':trophy:',
            'failure': ':warning:',
            'milestone': ':chart_with_upwards_trend:'
        }
        emoji = emoji_map.get(notification_type, ':gear:')
        
        text = f"{emoji} Training {notification_type.title()}: *{model_name}*"
        
        fields = [
            {
                "title": "Status",
                "value": training_data.get('status', 'unknown'),
                "short": True
            }
        ]
        
        if training_data.get('accuracy'):
            fields.append({
                "title": "Accuracy",
                "value": f"{training_data['accuracy']:.3f}",
                "short": True
            })
        
        if training_data.get('loss'):
            fields.append({
                "title": "Loss",
                "value": f"{training_data['loss']:.4f}",
                "short": True
            })
        
        if training_data.get('duration'):
            fields.append({
                "title": "Duration",
                "value": training_data['duration'],
                "short": True
            })
        
        color = '#36a64f' if notification_type == 'completion' else '#ffaa00'
        if notification_type == 'failure':
            color = '#ff0000'
        
        return {
            "text": text,
            "attachments": [
                {
                    "color": color,
                    "fields": fields,
                    "footer": "SNN-Fusion Training",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }
    
    def _format_alert_message(self, alert_data: Dict[str, Any], severity: str) -> Dict[str, Any]:
        """Format alert message."""
        title = alert_data.get('title', 'System Alert')
        message = alert_data.get('message', 'No details provided')
        
        emoji_map = {
            'info': ':information_source:',
            'warning': ':warning:',
            'error': ':exclamation:',
            'critical': ':rotating_light:'
        }
        emoji = emoji_map.get(severity, ':gear:')
        
        color_map = {
            'info': '#3AA3E3',
            'warning': '#ffaa00',
            'error': '#ff0000',
            'critical': '#8B0000'
        }
        color = color_map.get(severity, '#808080')
        
        text = f"{emoji} *{severity.upper()}*: {title}"
        
        return {
            "text": text,
            "attachments": [
                {
                    "color": color,
                    "text": message,
                    "footer": "SNN-Fusion Alert System",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }


class NotificationManager:
    """
    Centralized notification manager for coordinating email and Slack notifications.
    """
    
    def __init__(
        self,
        email_service: Optional[EmailNotificationService] = None,
        slack_service: Optional[SlackNotificationService] = None,
    ):
        """
        Initialize notification manager.
        
        Args:
            email_service: Email notification service instance
            slack_service: Slack notification service instance
        """
        self.email_service = email_service or EmailNotificationService()
        self.slack_service = slack_service or SlackNotificationService()
        self.logger = logging.getLogger(__name__)
    
    def notify_experiment_event(
        self,
        experiment_data: Dict[str, Any],
        event_type: str,
        recipients: Optional[List[str]] = None,
        slack_channel: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Send notifications for experiment events via all configured channels.
        
        Args:
            experiment_data: Experiment information
            event_type: Type of event (created, started, completed, failed)
            recipients: Email recipients (optional)
            slack_channel: Slack channel (optional)
            
        Returns:
            Success status for each notification channel
        """
        results = {}
        
        # Send email notification
        if recipients:
            results['email'] = self.email_service.send_experiment_notification(
                recipients=recipients,
                experiment_data=experiment_data,
                notification_type=event_type
            )
        
        # Send Slack notification
        results['slack'] = self.slack_service.send_experiment_update(
            experiment_data=experiment_data,
            update_type=event_type,
            channel=slack_channel
        )
        
        return results
    
    def notify_training_event(
        self,
        training_data: Dict[str, Any],
        event_type: str,
        recipients: Optional[List[str]] = None,
        slack_channel: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Send notifications for training events via all configured channels.
        
        Args:
            training_data: Training run information
            event_type: Type of event (started, completion, failure, milestone)
            recipients: Email recipients (optional)
            slack_channel: Slack channel (optional)
            
        Returns:
            Success status for each notification channel
        """
        results = {}
        
        # Send email notification
        if recipients:
            results['email'] = self.email_service.send_training_alert(
                recipients=recipients,
                training_data=training_data,
                alert_type=event_type
            )
        
        # Send Slack notification
        results['slack'] = self.slack_service.send_training_notification(
            training_data=training_data,
            notification_type=event_type,
            channel=slack_channel
        )
        
        return results