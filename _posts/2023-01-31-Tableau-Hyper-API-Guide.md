---
title: Tableau Hyper API Guide
description: 'Tableau .hyper 파일 CUID(Create, Update, Insert, Delete) 진행'
tags:
  - Tableau
  - API
categories:
  - STUDY
published: true
---

# What is Hyper API?

**Hyper API** 에는 Tableau 추출 파일과의 상호작용 및 자동화하는 데 사용할 수 있는 기능들이 포함되어 있습니다. Hyper API 를 사용하여 새 추출 파일을 생성하거나 기존 파일을 연 다음 해당 파일에서 테이블 생성(Create), 데이터 삽입(Insert), 삭제(Delete), 갱신(Update)하고 읽을 수 있습니다. Tableau Hyper API를 활용하기 위해 여타 블로그를 구글링했지만, 한국어로 된 포스트를 찾지 못해서 이번 기회에 Hyper API에 대한 가이드를 포스팅하게 되었습니다.

사용자는 Hyper API를 사용하여 아래의 기능을 수행할 수 있습니다.

- 태블로에서 지원하지 않는 데이터 원본에 대한 추출 파일 생성
- 사용자 지정 ETL(추출, 변환, 적재) 프로세스 자동화
- 추출 파일에서 데이터 검색

![](https://velog.velcdn.com/images/adastra/post/bfe4bb60-291d-4d14-abfe-7af93cfa5bc7/image.png)


Hyper API 는 아래와 같이 프로세스가 구성되어 있습니다.

![](https://velog.velcdn.com/images/adastra/post/e07efded-8bf1-4418-ad49-f6260f899e04/image.png)


1. Import: Hyper API 라이브러리 가져오기  
Hyper API를 사용하기 위해 필요한 라이브러리를 불러옵니다.

2. Start: Hyper API 프로세스 시작  
라이브러리를 불러온 후, Hyper 데이터베이스 서버를 시작합니다.

3. Open: Hyper API 파일에 대한 연결  
연결 개체를 사용하여 경로에 해당되는 .hyper 파일을 연결합니다.

4. Create: 테이블 생성 및 데이터 조작  
클래스를 사용하여 테이블을 정의 및 지정할 수 있고, 데이터를 삽입, 삭제, 갱신, 읽기를 수행할 수 있습니다.

# Hyper API 설치 및 라이브러리 준비

Hyper API는 Python, C++, Java 등 다양한 프로그래밍 언어로 패키지를 사용할 수 있습니다.  
해당 자료에서는 Python을 통해 Hyper API를 사용하고, 테스트 목적이므로 Colab를 사용하였습니다.  
*Colab: 웹 브라우저에서 주피터 노트북 에디터 형식 기반 텍스트와 코드를 자유롭게 작성할 수 있는 온라인 에디터 툴

구글 코랩 사이트를 접속하거나 해당 코드를 저장할 구글드라이브 폴더에서 우클릭 후, Google Colaboratory 를 클릭합니다.  
메뉴에서 보이지 않는 경우 하단의 연결할 앱 더보기를 클릭하여 Colaboratory를 설치합니다.

![](https://velog.velcdn.com/images/adastra/post/5420471c-7fb8-4981-bdf1-a99e490c977a/image.png)


![](https://velog.velcdn.com/images/adastra/post/4d456293-2ae9-49ca-a51a-816fe49da620/image.png)

1. Tableau Hyper API 를 설치합니다.

```powershell
pip install tableauhyperapi
```

Successfully installed tableauhyperapi 이 뜨면 성공적으로 설치된 것입니다.

2. API 설치 후, 필요한 라이브러리를 불러옵니다.(Colab 에는 pandas, numpy 등 기본 라이브러리가 내장되어 있으므로 별도로 설치할 필요가 없습니다.)

```python
import pandas as pd
import numpy as np
import shutil
```

```python
from datetime import datetime
from pathlib import Path

# 필요한 라이브러리 import 
from tableauhyperapi import HyperProcess, Telemetry, \
    Connection, CreateMode, \
    NOT_NULLABLE, NULLABLE, SqlType, TableDefinition, \
    Inserter, \
    escape_name, escape_string_literal, \
    HyperException
```

# Definition Table

Hyper API 실습에 사용할 테이블을 정의 및 생성하겠습니다.

이번 실습 목적으로 생성할 테이블은 “Orders”, “Customer”, “Products”, ”Line Items”, “test Items” 총 5개의 테이블입니다.

TableDefinition 메서드를 사용하여 테이블 정의를 만들고, 아래와 같이 테이블 명을 지정합니다.  이후 테이블 컬럼 명, 데이터 타입, null ability 등을 설정합니다. Orders 테이블 외 나머지 테이블도 아래와 같이 생성합니다.  
(Orders 테이블 정의와 동일한 형식이므로 코드는 생략하였습니다.)

```python
# 테이블 정의
orders_table = TableDefinition(

    # 테이블 명 지정
    table_name="Orders",

    columns=[
        TableDefinition.Column(name="Address ID", type=SqlType.small_int(), nullability=NOT_NULLABLE),
        TableDefinition.Column(name="Customer ID", type=SqlType.text(), nullability=NOT_NULLABLE),
        TableDefinition.Column(name="Order Date", type=SqlType.date(), nullability=NOT_NULLABLE),
        TableDefinition.Column(name="Order ID", type=SqlType.text(), nullability=NOT_NULLABLE),
        TableDefinition.Column(name="Ship Date", type=SqlType.date(), nullability=NULLABLE),
        TableDefinition.Column(name="Ship Mode", type=SqlType.text(), nullability=NULLABLE)
		]
)

customer_table = TableDefinition(
    
    table_name="Customer",

    columns=[
        TableDefinition.Column(name="Customer ID", type=SqlType.text(), nullability=NOT_NULLABLE),
        TableDefinition.Column(name="Customer Name", type=SqlType.text(), nullability=NOT_NULLABLE),
        TableDefinition.Column(name="Loyalty Reward Points", type=SqlType.big_int(), nullability=NOT_NULLABLE),
        TableDefinition.Column(name="Segment", type=SqlType.text(), nullability=NOT_NULLABLE)
    ]
)
```

# Insert Table

데이터 삽입을 위해 데이터베이스에 연결하고 연결된 데이터베이스에서 이전에 정의한 테이블 함수와 연결합니다.

삽입된 .hyper 파일은 superstore라는 이름으로 디렉토리에 저장됩니다.

```python
# data insert(데이터 삽입)함수
def run_insert_data_into_multiple_tables():

    print("EXAMPLE - Insert data into multiple tables within a new Hyper file")
    path_to_database = Path("superstore.hyper")

    with HyperProcess(telemetry=Telemetry.SEND_USAGE_DATA_TO_TABLEAU) as hyper:

        
        
        with Connection(endpoint=hyper.endpoint,
                        database=path_to_database,
                        create_mode=CreateMode.CREATE_AND_REPLACE) as connection:

            connection.catalog.create_table(table_definition=orders_table)
            connection.catalog.create_table(table_definition=customer_table)
            connection.catalog.create_table(table_definition=products_table)
            connection.catalog.create_table(table_definition=line_items_table)
            connection.catalog.create_table(table_definition=test_table)
```

Orders 테이블에 데이터를 생성하여 삽입합니다.  (이전에 열에서 지정한 데이터타입과 동일한 타입의 데이터를 입력해야 합니다.)

```python
# Insert data into Orders table.
orders_data_to_insert = [
     [399, "DK-13375", datetime(2012, 9, 7), "CA-2011-100006", datetime(2012, 9, 13), "Standard Class"],
     [530, "EB-13705", datetime(2012, 7, 8), "CA-2011-100090", datetime(2012, 7, 12), "Standard Class"],
     [777, "SM-24680", datetime(2013, 3, 2), "CA-2011-100099", datetime(2012, 3, 12), "Standard Class"]
]

with Inserter(connection, orders_table) as inserter:
                inserter.add_rows(rows=orders_data_to_insert)
                inserter.execute()
```

# Update Table

Hyper API를 이용하여 특정 조건을 지정한 후, .hyper 데이터를 조건에 따라 갱신할 수 있습니다.

superstore.hyper 파일을 데이터 원본으로 사용하여 업데이트 후 데이터를 반영할 super_store_sample_update.hyper 데이터파일을 만듭니다.

```python
# data update(조건에 맞는 데이터 검색하여 업데이트)함수
def run_update_data_into_multiple_tables():
    
    print("EXAMPLE_2 - Update data into multiple tables within a new Hyper file")

    # 데이터 원본은 superstore.hyper 사용
    path_to_source_database = Path("superstore.hyper")

    # superstore_sample_update.hyper 라는 파일 명으로 복사(shutil.copy)
    path_to_database = Path(shutil.copy(path_to_source_database, "superstore_sample_update.hyper")).resolve()
```

업데이트 전후를 확인하기 위해 갱신 전 Orders 테이블의 데이터를 확인합니다.

```python
# 업데이트 하기전 해당 테이블의 데이터 확인
rows_pre_update = connection.execute_list_query(
		query=f"SELECT {escape_name('Order Date')}, {escape_name('Order ID')}"
		f"FROM {escape_name('Orders')}")
print(f"Pre-Update: Individual rows showing 'Order Date' and 'Order ID' "
		f"columns: {rows_pre_update}\n")
```

![](https://velog.velcdn.com/images/adastra/post/fdd0e51b-c10b-44b5-ba83-01922c6f6a45/image.png)


Orders 테이블의 Order Date가 2012-10-01 이하거나 2013-03-01 이상에 해당되는 행의 날짜에 +10을 하는 쿼리 조건을 설정합니다.

```python
row_count = connection.execute_command(
		command=f"UPDATE {escape_name('Orders')} "
		f"SET {escape_name('Order Date')} = {escape_name('Order Date')} + 10 "
		f"WHERE {escape_name('Order Date')} <= '2012-08-01' OR {escape_name('Order Date')} >= '2013-03-01'")
```

Orders 테이블에서 갱신된 행의 갯수를 출력합니다.

```python
print(f"The number of updated rows in table {escape_name('Orders')} is {row_count}")
```

갱신한 후, 데이터를 확인합니다.

```python
rows_post_update = connection.execute_list_query(
		query=f"SELECT {escape_name('Order Date')}, {escape_name('Order ID')} "
		f"FROM {escape_name('Orders')}")
print(f"Post-Update: Individual rows showing 'Order Date' and 'Order ID'"
			f"columns: {rows_post_update}")
```

![](https://velog.velcdn.com/images/adastra/post/d9b0bd95-adec-4e52-8da6-436b089f49c2/image.png)


위 결과를 보면 조건에 해당하는 컬럼의 날짜가 +10이 된 것을 확인할 수 있습니다.

# Delete Table

이번엔 조건에 맞는 데이터를 삭제하는 프로세스를 진행하겠습니다.

superstore.hyper 파일을 데이터 원본으로 사용하여 업데이트 후 데이터를 반영할 super_store_sample_delete.hyper 데이터파일을 만듭니다.

```python
path_to_source_database = "superstore.hyper"

path_to_database = Path(shutil.copy(path_to_source_database, "superstore_sample_delete.hyper")).resolve()
```

Orders 테이블의 Order Date가 2012-08-01 이하에 해당되는 Customer ID를 Customer 테이블에서 삭제합니다.

```python
row_count = connection.execute_command(
		command=f"DELETE FROM {escape_name('Customer')} "
		f"WHERE {escape_name('Customer ID')} = ANY("
		f"SELECT {escape_name('Customer ID')} FROM {escape_name('Orders')} "
		f"WHERE {escape_name('Order Date')} <= '2012-08-01')")
```

![](https://velog.velcdn.com/images/adastra/post/6e175072-6965-4e6c-8bfc-248f4bd9fac5/image.png)


위와 같이 Orders 테이블에서 2012-08-01 이하에 해당되는 Customer ID의 행이 Customer 테이블에서 삭제된 것을 확인할 수 있습니다.

# Function

Insert, Update, Delete 등 데이터 조작을 위해 작성한 코드는 함수화한 다음 아래와 같이 실행해야 반영됩니다.

```python
# Function
if __name__ == '__main__':
    try:
        run_insert_data_into_multiple_tables()
        run_update_data_into_multiple_tables()
        run_delete_data_in_existing_hyper_file()
    except HyperException as ex:
        print(ex)
        exit(1)
```

이처럼 Hyper API는 태블로 데스크탑에서 .hyper 파일을 핸들링하는 목적으로 개발된 API입니다.

태블로 데스크탑이 아닌 태블로 서버에서 데이터 핸들링이 가능한 TSC(Tableau Server Client) 라이브러리도 존재합니다.  
해당 내용은 추후에 다루도록 하겠습니다.


*블로그에 있는 데이터프레임 형식의 사진은 .hyper 파일을 데이터프레임형식으로 변환해주는 Pantab이라는 라이브러리를 사용했습니다.




# Reference

[Tableau Hyper API 공식문서](https://help.tableau.com/current/api/hyper_api/en-us/index.html)  
[Tableau Hyper API 샘플코드(Github)](https://github.com/tableau/hyper-api-samples/tree/main/Tableau-Supporte)
