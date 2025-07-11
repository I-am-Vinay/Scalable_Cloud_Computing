from kafka import KafkaConsumer
from textblob import TextBlob
from collections import Counter, deque
import gender_guesser.detector as gender
import time
import psutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import boto3

consumer = KafkaConsumer(
    'amazon-fashion-reviews',
    bootstrap_servers='3.86.44.25:9092',
    auto_offset_reset='earliest',
    group_id='stream-group'
)

print("*** Kafka consumer connected. Streaming messages...")

d = gender.Detector()

workloads = [100000, 300000, 500000]
interval = 10000

metrics_records = []

for workload in workloads:
    print(f"\n========== Processing Workload {workload} ==========")

    sentiment_counts = Counter()
    gender_counts = Counter()
    name_window = deque(maxlen=interval * 3)  # rolling window for top names

    record_count = 0
    workload_start_time = time.time()
    interval_start_time = time.time()

    while record_count < workload:
        message = next(consumer)
        message_value = message.value.decode('utf-8')

        if ':::' in message_value:
            review_text, reviewer_name = message_value.split(':::', 1)
        else:
            continue

        analysis = TextBlob(str(review_text))
        polarity = analysis.sentiment.polarity
        sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
        sentiment_counts[sentiment] += 1

        if reviewer_name and reviewer_name.strip():
            first_name = reviewer_name.split()[0]
            gender_label = d.get_gender(first_name)
            name_window.append(first_name)
        else:
            gender_label = 'unknown'
        gender_counts[gender_label] += 1

        record_count += 1

        if record_count % interval == 0 or record_count == workload:
            batch_time = time.time() - interval_start_time
            throughput = interval / batch_time
            cpu_usage = psutil.cpu_percent(interval=None)
            mem_usage = psutil.virtual_memory().percent

            print(f"Processed {record_count} records — Batch Time: {batch_time:.2f}s, "
                  f"Throughput: {throughput:.2f}/sec, CPU: {cpu_usage}%, Mem: {mem_usage}%")

            metrics_records.append({
                'workload': workload,
                'records_processed': record_count,
                'time': batch_time,
                'throughput': throughput,
                'cpu': cpu_usage,
                'memory': mem_usage
            })

            interval_start_time = time.time()

    workload_total_time = time.time() - workload_start_time

    workload_metrics = [m for m in metrics_records if m['workload'] == workload]
    avg_time = sum(m['time'] for m in workload_metrics) / len(workload_metrics)
    avg_throughput = sum(m['throughput'] for m in workload_metrics) / len(workload_metrics)
    avg_cpu = sum(m['cpu'] for m in workload_metrics) / len(workload_metrics)
    avg_mem = sum(m['memory'] for m in workload_metrics) / len(workload_metrics)

    print(f"\n======> Final Metrics Summary for Workload {workload}:")
    print(f"   Total Records Processed: {workload}")
    print(f"   Total Workload Processing Time: {workload_total_time:.2f} seconds")
    print(f"   Average Time per Interval: {avg_time:.2f} seconds")
    print(f"   Average Throughput per Interval: {avg_throughput:.2f} records/sec")
    print(f"   Average CPU Usage per Interval: {avg_cpu:.2f}%")
    print(f"   Average Memory Usage per Interval: {avg_mem:.2f}%")
    print(f"\n*** Final Sentiment Counts: {dict(sentiment_counts)}")
    print(f"*** Final Gender Counts: {dict(gender_counts)}")

    # Print Top 5 Names from rolling window
    top_5_names = Counter(name_window).most_common(5)
    print(f"\n*** Top 5 Names (Rolling Window) after Workload {workload}:")
    for name, count in top_5_names:
        print(f"   {name}: {count} times")

metrics_df = pd.DataFrame(metrics_records)

# Metrics plotting function
def plot_metrics(df):
    sns.set(style='whitegrid')
    os.makedirs('results', exist_ok=True)
    palette = sns.color_palette("tab10", n_colors=len(workloads))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    metric_names = ['Time (s)', 'Throughput (records/sec)', 'CPU (%)', 'Memory (%)']
    columns = ['time', 'throughput', 'cpu', 'memory']

    for ax, col, name in zip(axes.flatten(), columns, metric_names):
        sns.lineplot(
            data=df,
            x='records_processed',
            y=col,
            hue='workload',
            ax=ax,
            marker='o',
            linewidth=2,
            markersize=8,
            palette=palette
        )
        ax.set_title(f"{name} vs Records Processed (Sequential Stream)", fontsize=16)
        ax.set_xlabel("Records Processed", fontsize=14)
        ax.set_ylabel(name, fontsize=14)
        ax.legend(title='Workload', fontsize=12)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('results/stream_sequential_results.png')
    plt.close()

    # Upload to S3
    local_path = 'results/stream_sequential_results.png'
    s3 = boto3.client('s3')
    bucket_name = 's3-bucket-x24112682'
    s3_key = 'results/stream_sequential_results.png'

    try:
        s3.upload_file(local_path, bucket_name, s3_key)
        print(f"Uploaded graph to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading graph to S3: {e}")

plot_metrics(metrics_df)

print("\n*** All combined graphs plotted and saved in 'results' folder.")
