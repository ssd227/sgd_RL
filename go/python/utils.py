import time

def test_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录函数开始时间
        result = func(*args, **kwargs)  # 调用原始函数
        end_time = time.time()  # 记录函数结束时间
        execution_time = end_time - start_time  # 计算函数执行时间
        print(f"{func.__name__} 执行时间: {execution_time:.6f} 秒")
        return result
    return wrapper