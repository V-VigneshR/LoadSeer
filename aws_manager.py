import boto3
import time
from datetime import datetime
from config import (
    region_name, 
    TARGET_GROUPS, 
    INSTANCE_IDS, 
    INSTANCE_START_TIMEOUT
)

# Initialize AWS clients
elbv2 = boto3.client('elbv2', region_name=region_name)
ec2 = boto3.client('ec2', region_name=region_name)

def wait_for_instance_state(instance_id, target_state='running'):
    start_time = time.time()
    while time.time() - start_time < INSTANCE_START_TIMEOUT:
        response = ec2.describe_instances(InstanceIds=[instance_id])
        state = response['Reservations'][0]['Instances'][0]['State']['Name']
        if state == target_state:
            return True
        time.sleep(5)
    return False

def manage_instance(operation, group):
    for instance_id in INSTANCE_IDS[group]:
        try:
            if operation == 'start':
                current_state = ec2.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]['State']['Name']
                if current_state not in ['running', 'pending']:
                    ec2.start_instances(InstanceIds=[instance_id])
                    print(f"Starting {group} instance {instance_id}")
                    if not wait_for_instance_state(instance_id):
                        raise Exception(f"Failed to start {group} instance {instance_id} within timeout")
                else:
                    print(f"Instance {instance_id} already {current_state}")
            elif operation == 'stop':
                ec2.stop_instances(InstanceIds=[instance_id])
                print(f"Stopping {group} instance {instance_id}")
        except Exception as e:
            print(f"Instance management error for {instance_id}: {str(e)}")
            raise

def register_instances(group):
    for instance_id in INSTANCE_IDS[group]:
        if not wait_for_instance_state(instance_id):
            raise Exception(f"Instance {instance_id} not in running state")

    targets = [{'Id': iid,'Port':8080} for iid in INSTANCE_IDS[group]]
    try:
        elbv2.register_targets(
            TargetGroupArn=TARGET_GROUPS[group],
            Targets=targets
        )
        print(f"Successfully registered {group} instances on port 8080")
    except Exception as e:
        print(f"Registration failed: {str(e)}")
        raise

def deregister_instances(group):
    try:
        targets = [{'Id': iid,'Port':8080} for iid in INSTANCE_IDS[group]]
        elbv2.deregister_targets(
            TargetGroupArn=TARGET_GROUPS[group],
            Targets=targets
        )
        print(f"Deregistered {group} instances from port 8080")
    except Exception as e:
        print(f"Deregistration error: {str(e)}")

def initialize_aws_resources():
    """Initialize AWS resources - start small instances, stop others"""
    manage_instance('start', 'small')
    register_instances('small')

    for group in ['medium', 'large']:
        manage_instance('stop', group)
        deregister_instances(group)