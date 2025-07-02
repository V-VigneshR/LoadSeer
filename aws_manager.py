# aws_manager.py
"""AWS EC2 and ELB management functionality"""

import boto3
import time
from config import AWS_REGION, TARGET_GROUPS, INSTANCE_IDS, INSTANCE_START_TIMEOUT


class AWSManager:
    """Manages AWS EC2 instances and load balancer operations"""
    
    def __init__(self):
        self.elbv2 = boto3.client('elbv2', region_name=AWS_REGION)
        self.ec2 = boto3.client('ec2', region_name=AWS_REGION)
        self.s3 = boto3.client('s3', region_name=AWS_REGION)

    def wait_for_instance_state(self, instance_id, target_state='running'):
        """Wait for instance to reach target state"""
        start_time = time.time()
        while time.time() - start_time < INSTANCE_START_TIMEOUT:
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            state = response['Reservations'][0]['Instances'][0]['State']['Name']
            if state == target_state:
                return True
            time.sleep(5)
        return False

    def manage_instance(self, operation, group):
        """Start or stop instances for a given group"""
        for instance_id in INSTANCE_IDS[group]:
            try:
                if operation == 'start':
                    current_state = self.ec2.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]['State']['Name']
                    if current_state not in ['running', 'pending']:
                        self.ec2.start_instances(InstanceIds=[instance_id])
                        print(f"Starting {group} instance {instance_id}")
                        if not self.wait_for_instance_state(instance_id):
                            raise Exception(f"Failed to start {group} instance {instance_id} within timeout")
                    else:
                        print(f"Instance {instance_id} already {current_state}")
                elif operation == 'stop':
                    self.ec2.stop_instances(InstanceIds=[instance_id])
                    print(f"Stopping {group} instance {instance_id}")
            except Exception as e:
                print(f"Instance management error for {instance_id}: {str(e)}")
                raise

    def register_instances(self, group):
        """Register instances with the load balancer"""
        for instance_id in INSTANCE_IDS[group]:
            if not self.wait_for_instance_state(instance_id):
                raise Exception(f"Instance {instance_id} not in running state")

        targets = [{'Id': iid, 'Port': 8080} for iid in INSTANCE_IDS[group]]
        try:
            self.elbv2.register_targets(
                TargetGroupArn=TARGET_GROUPS[group],
                Targets=targets
            )
            print(f"Successfully registered {group} instances on port 8080")
        except Exception as e:
            print(f"Registration failed: {str(e)}")
            raise

    def deregister_instances(self, group):
        """Deregister instances from the load balancer"""
        try:
            targets = [{'Id': iid, 'Port': 8080} for iid in INSTANCE_IDS[group]]
            self.elbv2.deregister_targets(
                TargetGroupArn=TARGET_GROUPS[group],
                Targets=targets
            )
            print(f"Deregistered {group} instances from port 8080")
        except Exception as e:
            print(f"Deregistration error: {str(e)}")

    def check_target_health(self, group):
        """Check health of targets in a group"""
        try:
            response = self.elbv2.describe_target_health(
                TargetGroupArn=TARGET_GROUPS[group],
                Targets=[{'Id': iid, 'Port': 8080} for iid in INSTANCE_IDS[group]]
            )
            return response['TargetHealthDescriptions'][0]['TargetHealth']['State'] == 'healthy'
        except Exception as e:
            print(f"Health check error for {group}: {str(e)}")
            return False
