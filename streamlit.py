import streamlit as st
import joblib 

#loader
model = joblib.load("C:/Users/Pc SToRe/Desktop/adam/work_from_home/model_work_from_home.pkl")
y = joblib.load("C:/Users/Pc SToRe/Desktop/adam/work_from_home/y (1).pkl")
lebel_day_type = joblib.load("C:/Users/Pc SToRe/Desktop/adam/work_from_home/day_type.pkl")
scaler = joblib.load("C:/Users/Pc SToRe/Desktop/adam/work_from_home/scaler.pkl")


st.set_page_config(page_title = "emplyee work from home" , page_icon= "👨‍💻")

st.title("Emplyee Work From Home 👨‍💻")

#day_type	work_hours	screen_time_hours	meetings_count	breaks_taken	after_hours_work	sleep_hours	task_completion_rate	burnout_score

day_type = st.selectbox("day_type:" , ["Weekday","Weekend"])
work_hours = st.number_input("work_hours" , 0.0 , 11.0 ,step = 1.0)
screen_time_hours =st.number_input("screen_time_hours" , 0.0 , 13.0 , step=1.0)
meetings_count= st.number_input("meetings_count" , 0 ,8)
breaks_taken = st.number_input("breaks_taken" , 0,5)
after_hours_work = st.selectbox("after_hours_work" , [0,1])
sleep_hours = st.number_input("sleep_hours" , 0.0,8.0 , step=1.0 )
task_completion_rate = st.number_input("task_completion_rate" , 0.0,100.0 , step=1.0)
burnout_score =st.number_input("burnout_score" , 0.0,150.0 ,step = 1.0)


day_type_lable = lebel_day_type.transform([day_type])[0]

input_scale = scaler.transform([[day_type_lable ,work_hours,screen_time_hours,meetings_count	,breaks_taken,after_hours_work,sleep_hours,task_completion_rate,burnout_score]])

if st.button("pred"):
    pred = model.predict(input_scale)
    output_text =y.inverse_transform(pred)[0]
    st.success(output_text)