## 1. scrape image
image file configuration
>images<br>
&emsp;-->face<br>
&emsp;&emsp;-->dilireba<br>
&emsp;&emsp;-->jiangwen<br>
&emsp;&emsp;-->pengyuyan<br>
&emsp;&emsp;&emsp;....
## 2. generate data list
>store the image paths and labels to data_list and dump the info to json file.

{<br>
&emsp; "all_class_images": 588,<br>
&emsp; "all_class_name": "face",<br>
&emsp; "all_class_sum": 6,<br>
&emsp; "class_detail": [<br>
&emsp;&emsp;         {<br>
&emsp;&emsp;&emsp;            "class_label": 0,<br>
&emsp;&emsp;&emsp;            "class_name": "dilireba",<br>
&emsp;&emsp;&emsp;            "class_test_images": 6,<br>
&emsp;&emsp;&emsp;            "class_trainer_images": 52<br>
&emsp;&emsp;        },<br>
&emsp;&emsp;        {<br>
&emsp;&emsp;&emsp;             "class_label": 1,<br>
&emsp;&emsp;&emsp;             "class_name": "jiangwen",<br>
&emsp;&emsp;&emsp;             "class_test_images": 11,
&emsp;&emsp;&emsp;             "class_trainer_images": 92<br>
&emsp;&emsp;        },<br>
&emsp;&emsp;        {<br>
&emsp;&emsp;&emsp;             "class_label": 2,<br>
&emsp;&emsp;&emsp;             "class_name": "pengyuyan",<br>
&emsp;&emsp;&emsp;             "class_test_images": 12,<br>
&emsp;&emsp;&emsp;             "class_trainer_images": 102<br>
&emsp;&emsp;        },<br>
&emsp;&emsp;        {<br>
&emsp;&emsp;&emsp;             "class_label": 3,<br>
&emsp;&emsp;&emsp;             "class_name": "tongliya",<br>
&emsp;&emsp;&emsp;             "class_test_images": 13,<br>
&emsp;&emsp;&emsp;             "class_trainer_images": 108<br>
&emsp;&emsp;        },<br>
&emsp;&emsp;        {<br>
&emsp;&emsp;&emsp;             "class_label": 4,<br>
&emsp;&emsp;&emsp;             "class_name": "wuyanzu",<br>
&emsp;&emsp;&emsp;             "class_test_images": 10,<br>
&emsp;&emsp;&emsp;             "class_trainer_images": 82<br>
&emsp;&emsp;        },<br>
&emsp;&emsp;        {<br>
&emsp;&emsp;&emsp;             "class_label": 5,<br>
&emsp;&emsp;&emsp;             "class_name": "zhangziyi",<br>
&emsp;&emsp;&emsp;             "class_test_images": 10,<br>
&emsp;&emsp;&emsp;             "class_trainer_images": 90<br>
&emsp;&emsp;        }<br>
&emsp;&emsp;    ]<br>
&emsp;}<br>
## 3. create image dataset.
>batch_data.py dump image data to file
## 4. train lenet model
>LeNet_keras.py
>load_data.py
## 5. predict image class