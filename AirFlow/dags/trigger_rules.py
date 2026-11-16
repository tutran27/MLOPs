# 3. Trigger Rules
# Mặc định, task chỉ chạy nếu toàn bộ upstream tasks thành công.

# Quy tắc này được gọi là: all_success.

# Nhưng có trường hợp bạn muốn một task chạy ngay cả khi upstream lỗi.
# Trigger Rule	Ý nghĩa
# all_success	Tất cả upstream thành công
# all_failed	Tất cả upstream thất bại
# all_done	Tất cả upstream đã kết thúc, bất kể kết quả
# one_failed	Có ít nhất một upstream thất bại
# none_failed_min_one_success	Không upstream nào failed và có ít nhất một success

from airflow.sdk import dag, task
from datetime import datetime


@dag(
    dag_id="trigger-rules",
    schedule=None,
    start_date=datetime(2026,11,16),
    catchup=False
)

def trigger_rules_dag():
    
    @task()
    def task_a():
        print("DONE: task_a")
    
    @task()
    def task_b():
        print("ERROR: task b")
        raise ValueError("task_b failed")
    
    @task()
    def task_c():
        print("DONE: task_c")
    
    @task(trigger_rule="all_done")
    def cleanup():
        print("DONE Cleaning")

    task_a() >> task_b() >> task_c() >> cleanup()

trigger_rules_dag()