import pandas as pd
from textblob import TextBlob
from multiprocessing import Pool, cpu_count
from collections import Counter
import time
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import gender_guesser.detector as gender
import os
import boto3
import warnings
warnings.filterwarnings("ignore")

def sentiment_and_gender_mapper(chunk): #mapper
    d = gender.Detector()
    sentiments = []
    genders = []
    for text, name in zip(chunk['reviewText'], chunk['reviewerName']):
        polarity = TextBlob(str(text)).sentiment.polarity
        sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
        sentiments.append(sentiment)
        if isinstance(name, str) and name.strip():
            first_name = name.split()[0]
            genders.append(d.get_gender(first_name))
        else:
            genders.append('unknown')
    return sentiments, genders

def process_interval_mapreduce(df_subset, pool): # mapper
    num_processes = cpu_count()
    chunk_size = int(len(df_subset) / num_processes)
    chunks = [df_subset.iloc[i:i + chunk_size] for i in range(0, len(df_subset), chunk_size)]

    start_time = time.time()
    cpu_usage = psutil.cpu_percent(interval=None)
    mem_usage = psutil.virtual_memory().percent

    results = pool.map(sentiment_and_gender_mapper, chunks)

    total_time = time.time() - start_time
    throughput = len(df_subset) / total_time

    sentiment_counts = Counter()
    gender_counts = Counter() # reducer
    for sentiments, genders in results:
        sentiment_counts.update(sentiments)
        gender_counts.update(genders)

    return total_time, throughput, cpu_usage, mem_usage, sentiment_counts, gender_counts

def plot_metrics(metrics_df, workloads):
    os.makedirs('results', exist_ok=True)
    sns.set(style='whitegrid')
    palette = sns.color_palette("tab10", n_colors=len(workloads))
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    metric_names = ['Time (s)', 'Throughput (reviews/sec)', 'CPU (%)', 'Memory (%)']
    columns = ['time', 'throughput', 'cpu', 'memory']

    for ax, col, name in zip(axes.flatten(), columns, metric_names):
        sns.lineplot(data=metrics_df, x='records_processed', y=col, hue='workload',
                     ax=ax, marker='o', markersize=8, linewidth=2, palette=palette)
        ax.set_title(f'{name} vs Records Processed', fontsize=16)
        ax.set_xlabel('Records Processed', fontsize=14)
        ax.set_ylabel(name, fontsize=14)
        ax.legend(title='Workload', fontsize=12)
        ax.grid(True)

    plt.tight_layout()
    local_path = 'results/mr_parallel_results.png'
    plt.savefig(local_path)
    plt.close()

    s3 = boto3.client('s3')
    bucket_name = 's3-bucket-x24112682'
    s3_key = 'results/mr_parallel_results.png'

    try:
        s3.upload_file(local_path, bucket_name, s3_key)
        print(f"Uploaded graph to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading graph to S3: {e}")

def main():
    s3 = boto3.client('s3')
    bucket_name = 's3-bucket-x24112682'
    s3_key = 'dataset/AMAZON_FASHION.json'
    local_file = 'AMAZON_FASHION.json'

    if not os.path.exists(local_file):
        print("Downloading dataset from S3...")
        s3.download_file(bucket_name, s3_key, local_file)
    else:
        print("Dataset already exists locally — skipping download.")

    df = pd.read_json(local_file, lines=True)
    workloads = [100000, 300000, 500000]
    interval = 10000
    metrics_records = []

    pool = Pool(processes=cpu_count())

    for workload in workloads:
        if workload <= len(df):
            print(f"\n========== Processing Workload {workload} ==========")
            workload_start_time = time.time()

            log_buffer = ""
            time_list, throughput_list, cpu_list, mem_list = [], [], [], []
            cumulative_sentiments, cumulative_genders = Counter(), Counter()

            for i in range(0, workload, interval):
                end = min(i + interval, workload)
                df_subset = df.iloc[i:end]

                total_time, throughput, cpu_usage, mem_usage, sentiment_counts, gender_counts = process_interval_mapreduce(df_subset, pool)

                time_list.append(total_time)
                throughput_list.append(throughput)
                cpu_list.append(cpu_usage)
                mem_list.append(mem_usage)

                cumulative_sentiments.update(sentiment_counts)
                cumulative_genders.update(gender_counts)

                log_buffer += (
                    f"Processed {end} records — Time: {total_time:.2f}s, "
                    f"Throughput: {throughput:.2f}/sec, CPU: {cpu_usage}%, Mem: {mem_usage}%\n"
                )

                metrics_records.append({
                    'workload': workload, 'records_processed': end,
                    'time': total_time, 'throughput': throughput, 'cpu': cpu_usage, 'memory': mem_usage
                })

            print(log_buffer)

            workload_total_time = time.time() - workload_start_time
            avg_time = sum(time_list) / len(time_list)
            avg_throughput = sum(throughput_list) / len(throughput_list)
            avg_cpu = sum(cpu_list) / len(cpu_list)
            avg_mem = sum(mem_list) / len(mem_list)

            print(f"\n======> Final Metrics Summary for Workload {workload}:")
            print(f"   Total Records Processed: {workload}")
            print(f"   Total Workload Processing Time: {workload_total_time:.2f} seconds")
            print(f"   Average Time per Interval: {avg_time:.2f} seconds")
            print(f"   Average Throughput per Interval: {avg_throughput:.2f} reviews/sec")
            print(f"   Average CPU Usage per Interval: {avg_cpu:.2f}%")
            print(f"   Average Memory Usage per Interval: {avg_mem:.2f}%")
            print(f"\n   Final Sentiment Counts: {dict(cumulative_sentiments)}")
            print(f"   Final Gender Counts: {dict(cumulative_genders)}\n")

        else:
            print(f"\n**** Skipping workload {workload} — dataset only has {len(df)} records.")

    pool.close()
    pool.join()

    metrics_df = pd.DataFrame(metrics_records)
    plot_metrics(metrics_df, workloads)

if __name__ == '__main__':
    main()
