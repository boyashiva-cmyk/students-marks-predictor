import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

def start_prediction_app():
    print("=========================================")
    print("   AI Student Final Marks Predictor")
    print("=========================================\n")

    try:
        df = pd.read_csv("student_data.csv")
    except FileNotFoundError:
        print("Error: student_data.csv kanapadatledu. File dhaggarlo undho ledo check chey.")
        return

    features = df[['Study_Hours', 'Previous_Scores']]
    target = df['Final_Marks']

    lr_model = LinearRegression()
    lr_model.fit(features, target)
    print("[SYSTEM] Model training complete! Ready for predictions.\n")

    while True:
        try:
            user_input_hours = input("Enter daily study hours (or type 'exit' to stop): ")
            
            if user_input_hours.lower() == 'exit':
                print("Exiting the predictor. Bye!")
                break
                
            hours = float(user_input_hours)
            prev_score = float(input("Enter previous test score (0-100): "))

            result = lr_model.predict([[hours, prev_score]])
            
            print("-----------------------------------------")
            print(f">>> Predicted Final Marks: {result[0]:.2f} / 100")
            print("-----------------------------------------\n")
            
        except ValueError:
            print("[ERROR] Please enter valid numbers only!\n")

if __name__ == "__main__":
    start_prediction_app()
