from kafka import KafkaConsumer
from textblob import TextBlob
from collections import Counter, deque
import gender_guesser.detector as gender
import time
import psutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count, freeze_support
import pandas as pd
import boto3

d = gender.Detector()

# Process each Kafka message
def process_message(message_value):
    if ':::' in message_value:
        review_text, reviewer_name = message_value.split(':::', 1)
    else:
        return ('Neutral', 'unknown', None)

    analysis = TextBlob(str(review_text))
    polarity = analysis.sentiment.polarity
    sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'

    if reviewer_name and reviewer_name.strip():
        first_name = reviewer_name.split()[0]
        gender_label = d.get_gender(first_name)
    else:
        gender_label = 'unknown'
        first_name = None

    return (sentiment, gender_label, first_name)

# Parallel consumer function
def run_parallel_consumer():
    consumer = KafkaConsumer(
        'amazon-fashion-reviews',
        bootstrap_servers='54.237.213.209:9092',
        auto_offset_reset='earliest',
        group_id='stream-parallel-group'
    )

    print("*** Parallel Kafka consumer connected.")

    workloads = [100000, 300000, 500000]
    interval = 10000
    num_processes = cpu_count()
    pool = Pool(num_processes)

    metrics_records = []

    for workload in workloads:
        print(f"\n========== Parallel Processing Workload {workload} ==========")

        sentiment_counts = Counter()
        gender_counts = Counter()
        name_window = deque(maxlen=interval * 3)  # window rolling

        total_messages = 0
        workload_start_time = time.time()
        messages_batch = []

        for message in consumer:
            messages_batch.append(message.value.decode('utf-8'))
            total_messages += 1

            if total_messages % interval == 0 or total_messages >= workload:
                batch_start = time.time()
                results = pool.map(process_message, messages_batch)
                batch_time = time.time() - batch_start

                for sentiment, gender_label, first_name in results:
                    sentiment_counts[sentiment] += 1
                    gender_counts[gender_label] += 1
                    if first_name:
                        name_window.append(first_name)

                throughput = len(messages_batch) / batch_time
                cpu_usage = psutil.cpu_percent(interval=None)
                mem_usage = psutil.virtual_memory().percent

                print(f"Processed {total_messages} records — Batch Time: {batch_time:.2f}s, "
                      f"Throughput: {throughput:.2f}/sec, CPU: {cpu_usage}%, Mem: {mem_usage}%")

                metrics_records.append({
                    'workload': workload,
                    'records_processed': total_messages,
                    'time': batch_time,
                    'throughput': throughput,
                    'cpu': cpu_usage,
                    'memory': mem_usage
                })

                messages_batch = []

            if total_messages >= workload:
                break

        workload_total_time = time.time() - workload_start_time

        avg_time = sum(m['time'] for m in metrics_records if m['workload'] == workload) / (workload // interval)
        avg_throughput = sum(m['throughput'] for m in metrics_records if m['workload'] == workload) / (workload // interval)
        avg_cpu = sum(m['cpu'] for m in metrics_records if m['workload'] == workload) / (workload // interval)
        avg_mem = sum(m['memory'] for m in metrics_records if m['workload'] == workload) / (workload // interval)

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

    pool.close()
    pool.join()

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
            ax.set_title(f"{name} vs Records Processed (Parallel Stream)", fontsize=16)
            ax.set_xlabel("Records Processed", fontsize=14)
            ax.set_ylabel(name, fontsize=14)
            ax.legend(title='Workload', fontsize=12)
            ax.grid(True)

        plt.tight_layout()
        plt.savefig('results/stream_parallel_results.png')
        plt.close()

        # Upload to S3
        local_path = 'results/stream_parallel_results.png'
        s3 = boto3.client('s3')
        bucket_name = 's3-bucket-x24112682'
        s3_key = 'results/stream_parallel_results.png'

        try:
            s3.upload_file(local_path, bucket_name, s3_key)
            print(f"Uploaded graph to s3://{bucket_name}/{s3_key}")
        except Exception as e:
            print(f"Error uploading graph to S3: {e}")

    plot_metrics(metrics_df)
    print("\n*** All combined graphs plotted and saved in 'results' folder.")

if __name__ == '__main__':
    freeze_support()
    run_parallel_consumer()
