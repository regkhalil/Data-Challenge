"""Local evaluation script for the Idemia Face Occlusion Challenge."""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import cv2


try:
    from submit_baseline import estimate_occlusion
except ImportError:
    print("Error: Could not import 'estimate_occlusion' from 'submit_baseline.py'.")
    print("Make sure both files are in the same directory.")
    sys.exit(1)

TRAIN_CSV = Path("occlusion_datasets/train.csv") 
IMAGE_DIR = Path("crops/Crop_224_5fp_100K")
SAMPLE_SIZE = 1000

def evaluate_local() -> None:
    if not TRAIN_CSV.is_file():
        print(f"Error: Could not find {TRAIN_CSV}. Please update the TRAIN_CSV path in the script.")
        return
        
    print(f"Loading training data from {TRAIN_CSV}...")

    df_train = pd.read_csv(TRAIN_CSV, delimiter=",").dropna(subset=["filename", "FaceOcclusion"])
    
    if len(df_train) > SAMPLE_SIZE:
        df_train = df_train.sample(n=SAMPLE_SIZE, random_state=42)
        
    records = []
    
    for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Evaluating", unit="img"):
        filename = row["filename"]
        gt_occlusion = row["FaceOcclusion"]
        
        image_path = IMAGE_DIR / filename
        if not image_path.is_file():
            continue
            
        pred_occlusion = estimate_occlusion(image_path)
        
        weight = (1.0 / 30.0) + gt_occlusion
        sq_error = (pred_occlusion - gt_occlusion) ** 2
        weighted_sq_error = weight * sq_error
        abs_error = abs(pred_occlusion - gt_occlusion)
        
        records.append({
            "filename": filename,
            "GT": gt_occlusion,
            "Prediction": pred_occlusion,
            "Weight": weight,
            "Weighted_Sq_Err": weighted_sq_error,
            "Abs_Error": abs_error
        })
        
    if not records:
        print("No images were successfully processed. Check your IMAGE_DIR path.")
        return
        
    results_df = pd.DataFrame(records)
    total_weighted_sq_err = results_df["Weighted_Sq_Err"].sum()
    total_weight = results_df["Weight"].sum()
    
    final_score = total_weighted_sq_err / total_weight
    
    print("\n" + "="*60)
    print(f"LOCAL EVALUATION RESULTS ({len(results_df)} images)")
    print("="*60)
    print(f"Idemia Challenge Metric Score: {final_score:.6f}")
    print("(Note: Lower is better. 0.0 is perfect.)\n")
    
    print("TOP 5 WORST PREDICTIONS (By Absolute Error):")
    worst_5 = results_df.nlargest(5, "Abs_Error")
    
    # Find the top 5 worst predictions
    print("TOP 5 WORST PREDICTIONS (By Absolute Error):")
    worst_5 = results_df.nlargest(5, "Abs_Error")
    
    for _, row in worst_5.iterrows():
        print(f"- File: {row['filename']}")
        print(f"  Predicted: {row['Prediction']:.4f} | True: {row['GT']:.4f} | Error Diff: {row['Abs_Error']:.4f}")
        print("-" * 60)

    print("\nOpening worst images... Press ANY KEY on the image window to see the next one (or 'q' to quit).")
    
    # Visual Debugging Loop
    for _, row in worst_5.iterrows():
        img_path = IMAGE_DIR / row['filename']
        img = cv2.imread(str(img_path))
        
        if img is not None:
            # Resize by 2x so it's easier to analyze on screen
            img_display = cv2.resize(img, (448, 448))
            
            # Add a black background bar for readable text
            cv2.rectangle(img_display, (0, 0), (448, 40), (0, 0, 0), -1)
            
            # Write the stats on the image (True GT vs our Prediction)
            text = f"True: {row['GT']:.2f} | Pred: {row['Prediction']:.2f}"
            cv2.putText(img_display, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the window and wait for the user to press a key
            cv2.imshow("Visual Debugger (Press any key for next)", img_display)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            
            # If the user presses 'q', exit the viewing loop early
            if key == ord('q'):
                break
        else:
            print(f"Could not load image to display: {img_path}")
if __name__ == "__main__":
    evaluate_local()