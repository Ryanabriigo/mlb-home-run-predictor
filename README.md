# MLB Daily Home Run Predictor

This is the one-click web-app version.

## Put it online
1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Open Streamlit Community Cloud and deploy `app.py`.
4. Bookmark the resulting app on your phone or add it to your Home Screen.
5. Each day tap **RUN TODAY'S PREDICTIONS**.

The app uses MLB StatsAPI for schedules/probable pitchers and Baseball Savant Statcast data through pybaseball.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
