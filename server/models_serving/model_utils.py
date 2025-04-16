# server/model_utils.py
def enhance_dicom(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)
    
    # Apply model-based enhancements
    enhanced = cv2.detailEnhance(
        img, 
        sigma_s=15, 
        sigma_r=0.15
    )
    
    # Overlay predictions
    overlay = generate_attention_overlay(img)
    blended = cv2.addWeighted(enhanced, 0.7, overlay, 0.3, 0)
    
    return blended
