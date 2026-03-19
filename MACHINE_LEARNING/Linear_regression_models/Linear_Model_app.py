import streamlit as st

from Linear_model_comparision import df_linear

st.write("###  LINEAR COMPARISION MODEL")
st.dataframe(df_linear)

best_linear = df_linear.iloc[0]
st.success(f"Best Linear Model: {best_linear['Model']} ")