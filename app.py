import streamlit as st

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def main():

    st.title('Mental Order Classification')

    sadness = st.selectbox('Choose Sadness level from option below', ['Usually','Sometimes' ,'Seldom' ,'Most-Often'])

    euphoric = st.selectbox('Choose Euphoric level from option below', ['Seldom' ,'Most-Often', 'Usually', 'Sometimes'])

    exhausted = st.selectbox('Choose Exhausted level from option below', ['Sometimes', 'Usually' ,'Seldom', 'Most-Often'])

    sleep_dissorder = st.selectbox('Choose Sleep Dissorder level from option below', ['Sometimes' ,'Most-Often' ,'Usually' ,'Seldom'])

    mood_swing = st.selectbox('Choose Mood Swing level from option below', ['YES','NO'])

    suicidal_thoughts = st.selectbox('Choose Suicidal Thoughts level from option below', ['YES' ,'NO'])

    anorxia = st.selectbox('Choose Anorxia level from option below', ['YES' ,'NO'])

    authority_respect = st.selectbox('Choose Authority Respect level from option below', ['YES' ,'NO'])

    try_explanation = st.selectbox('Choose Try Explanation level from option below', ['YES', 'NO'])

    aggressive_response = st.selectbox('Choose Aggressive Response level from option below', ['YES' ,'NO'])

    ignore_move_on = st.selectbox('Choose Ignore and move on level from option below', ['YES', 'NO'])

    nervous_break_down = st.selectbox('Choose Nervous break down level from option below', ['YES' ,'NO'])

    admit_mistakes = st.selectbox('Choose Admit Mistakes level from option below', ['YES', 'NO'])

    overthinking = st.selectbox('Choose Overthinking level from option below', ['YES' ,'NO'])

    sexual_activity = st.number_input('Enter Sexual Activity ', value= 0, min_value= 0, max_value= 10)

    concentration = st.number_input('Enter Concentration ', value= 0, min_value= 0, max_value= 10)

    optimisim = st.number_input('Enter Optimisim level', value= 0, min_value= 0, max_value= 10)

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
        label= 'Predict Diagnose'
    )

    if button:

        pred_df=data.get_data_as_dataframe()

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)

        st.write(f'You are diagnosed with {results[0]}')




if __name__ == '__main__':

    main()