import pandas as pd
from textblob import TextBlob
from collections import Counter
import time
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import gender_guesser.detector as gender
import os
import boto3

def sentiment_analysis_sequential(df): #mapper
    sentiments = []
    for text in df['reviewText']:
        analysis = TextBlob(str(text))
        polarity = analysis.sentiment.polarity
        sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
        sentiments.append(sentiment)
    return sentiments

def gender_detection_sequential(df): #mapper
    d = gender.Detector()
    genders = []
    for name in df['reviewerName']:
        if isinstance(name, str) and name.strip():
            first_name = name.split()[0]
            genders.append(d.get_gender(first_name))
        else:
            genders.append('unknown')
    return genders

def process_workload_sequential(df, workload_size, interval):
    interval_times, throughputs, cpu_usages, memory_usages, record_counts = [], [], [], [], []

    cumulative_sentiment_counts = Counter()
    cumulative_gender_counts = Counter()

    for i in range(0, workload_size, interval):
        end = min(i + interval, workload_size)
        df_subset = df.iloc[i:end]

        start_time = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        mem_usage = psutil.virtual_memory().percent

        sentiments = sentiment_analysis_sequential(df_subset)
        genders = gender_detection_sequential(df_subset)

        cumulative_sentiment_counts.update(sentiments) #reducer
        cumulative_gender_counts.update(genders) #reducer

        total_time = time.time() - start_time
        throughput = len(df_subset) / total_time

        interval_times.append(total_time)
        throughputs.append(throughput)
        cpu_usages.append(cpu_usage)
        memory_usages.append(mem_usage)
        record_counts.append(end)

        print(f"Processed {end} records — Time: {total_time:.2f}s, Throughput: {throughput:.2f}/sec, CPU: {cpu_usage}%, Mem: {mem_usage}%")

    return record_counts, interval_times, throughputs, cpu_usages, memory_usages, cumulative_sentiment_counts, cumulative_gender_counts

def plot_metrics(metrics_df, workloads):
    os.makedirs('results', exist_ok=True)

    sns.set(style='whitegrid')
    palette = sns.color_palette("tab10", n_colors=len(workloads))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    metric_names = ['Time (s)', 'Throughput (reviews/sec)', 'CPU (%)', 'Memory (%)']
    columns = ['time', 'throughput', 'cpu', 'memory']

    for ax, col, name in zip(axes.flatten(), columns, metric_names):
        sns.lineplot(
            data=metrics_df,
            x='records_processed',
            y=col,
            hue='workload',
            ax=ax,
            marker='o',
            markersize=8,
            linewidth=2,
            palette=palette
        )
        ax.set_title(f'{name} vs Records Processed', fontsize=16)
        ax.set_xlabel('Records Processed', fontsize=14)
        ax.set_ylabel(name, fontsize=14)
        ax.legend(title='Workload', fontsize=12)
        ax.grid(True)

    plt.tight_layout()
    local_path = 'results/mr_sequential_results.png'
    plt.savefig(local_path)
    plt.close()

    # Upload to S3
    s3 = boto3.client('s3')
    bucket_name = 's3-bucket-x24112682'
    s3_key = 'results/mr_sequential_results.png'

    try:
        s3.upload_file(local_path, bucket_name, s3_key)
        print(f"ploaded graph to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading graph to S3: {e}")


def main():
    #df = pd.read_json('Dataset/AMAZON_FASHION.json', lines=True)
    
    s3 = boto3.client('s3')
    bucket_name = 's3-bucket-x24112682' 
    s3_key = 'dataset/AMAZON_FASHION.json'
    local_file = 'AMAZON_FASHION.json'

    # Download from S3 to local temp file (if not already present)
    if not os.path.exists(local_file):
        print("Downloading dataset from S3...")
        s3.download_file(bucket_name, s3_key, local_file)
    else:
        print("Dataset already exists locally — skipping download.")

    # Load dataset
    df = pd.read_json(local_file, lines=True)

    workloads = [100000, 300000, 500000]
    interval = 10000

    metrics_records = []

    for workload in workloads:
        if workload <= len(df):
            print(f"\n========== Processing Workload {workload} ==========")

            workload_start_time = time.time()

            rec_counts, interval_times, throughputs, cpus, mems, sentiment_counts, gender_counts = process_workload_sequential(df, workload, interval)

            for rec, t, thr, c, m in zip(rec_counts, interval_times, throughputs, cpus, mems):
                metrics_records.append({
                    'workload': workload,
                    'records_processed': rec,
                    'time': t,
                    'throughput': thr,
                    'cpu': c,
                    'memory': m
                })

            workload_total_time = time.time() - workload_start_time
            avg_time = sum(interval_times) / len(interval_times)
            avg_throughput = sum(throughputs) / len(throughputs)
            avg_cpu = sum(cpus) / len(cpus)
            avg_mem = sum(mems) / len(mems)

            print(f"\n======> Final Metrics Summary for Workload {workload}:")
            print(f"   Total Records Processed: {workload}")
            print(f"   Total Workload Processing Time: {workload_total_time:.2f} seconds")
            print(f"   Average Time per Interval: {avg_time:.2f} seconds")
            print(f"   Average Throughput per Interval: {avg_throughput:.2f} reviews/sec")
            print(f"   Average CPU Usage per Interval: {avg_cpu:.2f}%")
            print(f"   Average Memory Usage per Interval: {avg_mem:.2f}%")
            print(f"\n   Final Sentiment Counts: {dict(sentiment_counts)}")
            print(f"   Final Gender Counts: {dict(gender_counts)}")

        else:
            print(f"\n**** Skipping workload {workload} — dataset only has {len(df)} records.")

    metrics_df = pd.DataFrame(metrics_records)
    plot_metrics(metrics_df, workloads)

if __name__ == '__main__':
    main()
