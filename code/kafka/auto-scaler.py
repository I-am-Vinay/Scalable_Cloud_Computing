import boto3
import time
from datetime import datetime, timedelta

# === Configuration Section ===

AMI_ID = 'ami-0f7a9d46859727c76'  
INSTANCE_TYPE = 't3.large'
SECURITY_GROUP_ID = 'sg-048641ac4359ccf08'
SUBNET_ID = 'subnet-04cc0e1c4a0f7f651'

CPU_HIGH_THRESHOLD = 60
CPU_LOW_THRESHOLD = 15
INSTANCE_LIMIT = 2
MONITOR_INTERVAL = 40  # in seconds

WORKER_TAG = {'Key': 'scalable-tag', 'Value': 'cloud'}

ec2_client = boto3.client('ec2')
cloudwatch_client = boto3.client('cloudwatch')


def list_active_instances():
    """Return list of active EC2 worker instance IDs for this project."""
    response = ec2_client.describe_instances(
        Filters=[
            {'Name': 'tag:scalable-tag', 'Values': [WORKER_TAG['Value']]},
            {'Name': 'instance-state-name', 'Values': ['running']}
        ]
    )
    instance_ids = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instance_ids.append(instance['InstanceId'])
    return instance_ids


def fetch_average_cpu(instance_ids):
    """Get average CPU utilization for given instances over the past 5 minutes."""
    if not instance_ids:
        return 0

    now = datetime.utcnow()
    past = now - timedelta(minutes=5)
    total_cpu = 0
    count = 0

    for instance_id in instance_ids:
        metrics = cloudwatch_client.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=past,
            EndTime=now,
            Period=300,
            Statistics=['Average']
        )
        datapoints = metrics.get('Datapoints', [])
        if datapoints:
            total_cpu += datapoints[0]['Average']
            count += 1

    return total_cpu / count if count else 0


def create_worker():
    """Launch a new EC2 instance and run stream.py."""
    print("****** Creating a new worker instance ****** ")
    try:
        response = ec2_client.run_instances(
            ImageId=AMI_ID,
            InstanceType=INSTANCE_TYPE,
            MinCount=1,
            MaxCount=3,
            SecurityGroupIds=[SECURITY_GROUP_ID],
            SubnetId=SUBNET_ID,
            UserData='''#!/bin/bash
            cd /home/ec2-user/environment/CODE/Kafka
            nohup python3 stream_parallel.py > stream.log 2>&1 &
            ''',
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [WORKER_TAG]
                }
            ]
        )
        new_instance_id = response['Instances'][0]['InstanceId']
        print(f"----> Worker instance {new_instance_id} launched.")
    except Exception as ex:
        print("----> Failed to start EC2 instance:", ex)





# === Auto-Scaling Control Loop ===

print("\n[Auto-Scaler] Stream workload manager active...\n")

while True:
    active_instances = list_active_instances()
    average_cpu = fetch_average_cpu(active_instances)
    print(f"Active Workers: {len(active_instances)} | Avg CPU: {average_cpu:.2f}%")

    if average_cpu > CPU_HIGH_THRESHOLD and len(active_instances) < INSTANCE_LIMIT:
        print("----> Detected High Workload. Adding one worker.")
        create_worker()

    elif average_cpu < CPU_LOW_THRESHOLD and len(active_instances) > 1:
        print("----> Detected Low workload. Removing one worker.")

    else:
        print("----> No need of scaling action needed.")

    time.sleep(MONITOR_INTERVAL)
