import streamlit as st
from model_comparision import df

st.title(" ML MODEL COMPARISION DASHBOARD")

st.write("### Model Ranking")

st.dataframe(df)

best_model = df.iloc[0]

st.success(f"Best Model: {best_model['Model']} ")

# Optional chart
st.bar_chart(df.set_index("Model")["R2"])