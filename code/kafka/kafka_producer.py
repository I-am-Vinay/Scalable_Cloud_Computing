from kafka import KafkaProducer
import pandas as pd
import time

df = pd.read_json('/home/ec2-user/environment/AMAZON_FASHION.json', lines=True)

producer = KafkaProducer(
    bootstrap_servers='54.237.213.209:9092',
    batch_size=16384,
    linger_ms=10,          
    buffer_memory=33554432
)

print("*** Kafka producer connected. Sending messages...")

# Track sending start time
start_time = time.time()
record_count = 0

# Send messages to 'tweets' topic
for index, row in df.iterrows():
    review_text = str(row['reviewText']) if 'reviewText' in row and pd.notnull(row['reviewText']) else ''
    reviewer_name = str(row['reviewerName']) if 'reviewerName' in row and pd.notnull(row['reviewerName']) else ''
    message = f"{review_text}:::{reviewer_name}"
    producer.send('amazon-fashion-reviews', value=message.encode('utf-8'))

    record_count += 1


# Finalize and flush
producer.flush()
producer.close()

elapsed_time = time.time() - start_time
throughput = record_count / elapsed_time

print(f"\n*** Sent {record_count} messages in {elapsed_time:.2f} seconds.")
print(f"*** Producer Throughput: {throughput:.2f} messages/sec")
