#Dockerfile, Image, Container

FROM python:3.11.1

ADD Ejemplo/main.py .
ADD Ejemplo/Position_Salaries.csv .

#RUN pip install sckit-learn
#RUN pip install sklearn
RUN pip install -U scikit-learn
RUN pip install pandas
RUN pip install streamlit
RUN pip install numpy
RUN pip install flask
RUN pip install flask-cors
RUN pip install matplotlib

CMD ["python","./main.py"]