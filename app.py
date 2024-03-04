import streamlit as st

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def main():

    st.title('Mental Order Classification')

    sadness = st.selectbox('Choose sadness', ['Usually','Sometimes' ,'Seldom' ,'Most-Often'])

    euphoric = st.selectbox('Choose euphoric', ['Seldom' ,'Most-Often', 'Usually', 'Sometimes'])

    exhausted = st.selectbox('Choose exhausted', ['Sometimes', 'Usually' ,'Seldom', 'Most-Often'])

    sleep_dissorder = st.selectbox('Choose sleep_dissorder', ['Sometimes' ,'Most-Often' ,'Usually' ,'Seldom'])

    mood_swing = st.selectbox('Choose mood_swing', ['YES','NO'])

    suicidal_thoughts = st.selectbox('Choose suicidal_thoughts', ['YES' ,'NO'])

    anorxia = st.selectbox('Choose anorxia', ['YES' ,'NO'])

    authority_respect = st.selectbox('Choose authority_respect', ['YES' ,'NO'])

    try_explanation = st.selectbox('Choose try_explanation', ['YES', 'NO'])

    aggressive_response = st.selectbox('Choose aggressive_response', ['YES' ,'NO'])

    ignore_move_on = st.selectbox('Choose ignore_move_on', ['YES', 'NO'])

    nervous_break_down = st.selectbox('Choose nervous_break_down', ['YES' ,'NO'])

    admit_mistakes = st.selectbox('Choose admit_mistakes', ['YES', 'NO'])

    overthinking = st.selectbox('Choose overthinking', ['YES' ,'NO'])

    sexual_activity = st.number_input('Enter sexual_activity', value= 0, min_value= 0, max_value= 10)

    concentration = st.number_input('Enter concentration', value= 0, min_value= 0, max_value= 10)

    optimisim = st.number_input('Enter optimisim', value= 0, min_value= 0, max_value= 10)

    data = CustomData(
        sadness= sadness,
        euphoric= euphoric,
        exhausted= exhausted,
        sleep_dissorder= sleep_dissorder,
        mood_swing= mood_swing,
        suicidal_thoughts= suicidal_thoughts,
        anorxia= anorxia,
        authority_respect= authority_respect,
        try_explanation= try_explanation,
        aggressive_response= aggressive_response,
        ignore_move_on= ignore_move_on,
        nervous_break_down= nervous_break_down,
        admit_mistakes= admit_mistakes,
        overthinking= overthinking,
        sexual_activity= sexual_activity,
        concentration= concentration,
        optimisim= optimisim
        
    )

    button = st.button(
        label= 'Predict Diesea'
    )

    if button:

        pred_df=data.get_data_as_dataframe()

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)

        st.write(f'{results[0]}')




if __name__ == '__main__':

    main()