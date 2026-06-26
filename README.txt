AgriCrop AI - Streamlit Community Cloud Deployment

Files included:
1. app.py
2. crop_recommendation_model.pkl
3. requirements.txt
4. sample_input.csv

Deployment:
1. Create a new public GitHub repository.
2. Upload these files to the repository root, not inside a folder.
3. Go to Streamlit Community Cloud.
4. Click Create app / New app.
5. Connect your GitHub account.
6. Select your repository.
7. Set main file path as app.py.
8. Deploy.

Model:
- Algorithm: Random Forest Classifier
- Target: crop label
- Inputs: N, P, K, temperature, humidity, ph, rainfall
- Test accuracy before final full-dataset retraining: 0.9932
