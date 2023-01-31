---
title: Tableau Hyper API Guide
description: Tableau .hyper 파일 CUID(Create, Update, Insert, Delete) 진행
tags:
- Tableau
- API
categories:
- STUDY
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

1. 먼저 구글 코랩 사이트를 접속하거나 해당 코드를 저장할 구글드라이브 폴더에서 우클릭 후

Google Colaboratory 를 클릭합니다.

메뉴에서 보이지 않는 경우 하단의 연결할 앱 더보기를 클릭하여 Colaboratory를 설치합니다.

![](https://velog.velcdn.com/images/adastra/post/5420471c-7fb8-4981-bdf1-a99e490c977a/image.png)


![](https://velog.velcdn.com/images/adastra/post/4d456293-2ae9-49ca-a51a-816fe49da620/image.png)

1. 먼저 Tableau Hyper API 를 설치합니다.

```powershell
pip install tableauhyperapi
```

Successfully installed tableauhyperapi 이 뜨면 성공적으로 설치된 것입니다.

1. API 설치 후, 필요한 라이브러리를 불러옵니다.(Colab 에는 pandas, numpy 등 기본 라이브러리가 내장되어 있으므로 별도로 설치할 필요가 없습니다.)

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

1. TableDefinition 메서드를 사용하여 테이블 정의를 만들고, 아래와 같이 테이블 명을 지정합니다.

이후 테이블 컬럼 명, 데이터 타입, null ability 등을 설정합니다.

Orders 테이블 외 나머지 테이블도 아래와 같이 생성합니다.

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

Orders 테이블에 데이터를 생성하여 삽입합니다.

(이전에 열에서 지정한 데이터타입과 동일한 타입의 데이터를 입력해야 합니다.)

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






참고자료

[Tableau Hyper API 공식문서](https://help.tableau.com/current/api/hyper_api/en-us/index.html)

[Tableau Hyper API 샘플코드(Github)](https://github.com/tableau/hyper-api-samples/tree/main/Tableau-Supported)