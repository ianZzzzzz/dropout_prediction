dtypes = {
    "enroll_id": "int64",
    "timestamp": "int64",
    "username": "int32",
    "content_id": "int16",
    "content_type_id": "boolean",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32", 
    "prior_question_had_explanation": "int8"
}
path = ''
train = cudf.read_csv(path, dtype=dtypes)

data_type = {
    'enroll_id':'int64'
    ,'username':'int64'
    ,'course_id':'str'
    ,'session_id':'str'
    ,'action':'str'
    ,'object':'str'
    ,'time':'str'}      