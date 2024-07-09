import streamlit as st
import numpy as np
import cv2
from streamlit_image_coordinates import streamlit_image_coordinates 

def image_formation_model(f, x0, y0, sigma):
    nr, nc = f.shape[:2]
    
    # Create coordinate grids
    x = np.arange(nr).reshape(-1, 1)
    y = np.arange(nc).reshape(1, -1)
    
    # Calculate the illumination matrix
    illumination = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    
    # Apply illumination to each color channel
    g = (illumination[..., np.newaxis] * f).astype(np.uint8)
    return g

def image_quantization(f, bits):
    levels = 2 ** bits
    interval = 256 / levels
    gray_level_interval = 255 / (levels - 1)
    
    # 使用 numpy 的向量化操作來創建查找表
    k = np.arange(256)
    l = np.floor(k / interval).astype(int)
    table = np.round(l * gray_level_interval).astype(np.uint8)
    
    return cv2.LUT(f, table)

def image_downsampling(f, sampling_rate):
    return f[::sampling_rate, ::sampling_rate]

def crop_image(image, x, y, width, height):
    return image[y:y+height, x:x+width]


def main():
    st.title('Image Processing App')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "bmp"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        st.image(img_array, caption='Uploaded Image', use_column_width=True)
        
        processing_option = st.selectbox(
            'Select processing option',
            ('Image Formation Model', 'Image Quantization', 'Image Downsampling', 'Image Cropping')
        )
        
        if processing_option == 'Image Formation Model':
            nr, nc = img_array.shape[:2]
            x0 = st.slider('X0', 0, nr, nr // 2)
            y0 = st.slider('Y0', 0, nc, nc // 2)
            sigma = st.slider('Sigma', 1, 500, 200)
            
            processed_img = image_formation_model(img_array, x0, y0, sigma)
            st.image(processed_img, caption='Processed Image', use_column_width=True)
        
        elif processing_option == 'Image Quantization':
            bits = st.slider('Bits', 1, 8, 8)
            
            processed_img = image_quantization(img_array, bits)
            st.image(processed_img, caption='Processed Image', use_column_width=True)
        
        elif processing_option == 'Image Downsampling':
            sampling_rate = st.slider('Sampling Rate', 2, 10, 2)
            
            processed_img = image_downsampling(img_array, sampling_rate)
            st.image(processed_img, caption='Processed Image', use_column_width=True)
        
        elif processing_option == 'Image Cropping':
            '''h, w = img_array.shape[:2]
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.number_input("X1", 0, w-1, 0)
                y1 = st.number_input("Y1", 0, h-1, 0)
            with col2:
                x2 = st.number_input("X2", x1+1, w, w)
                y2 = st.number_input("Y2", y1+1, h, h)
            
            cropped_img = crop_image(img_array, x1, y1, x2-x1, y2-y1)
            st.image(cropped_img, caption='Cropped Image', use_column_width=True)'''
            
            crop_method = st.radio("Select cropping method", ("Auto", "Manual"))
            
            if crop_method == "Auto":
                target_width = st.number_input("Target Width", min_value=1, max_value=img_array.shape[1], value=100)
                target_height = st.number_input("Target Height", min_value=1, max_value=img_array.shape[0], value=100)
                
                if st.button('Crop'):
                    h, w = img_array.shape[:2]
                    start_x = (w - target_width) // 2
                    start_y = (h - target_height) // 2
                    cropped_img = crop_image(img_array, start_x, start_y, target_width, target_height)
                    st.image(cropped_img, caption='Cropped Image', use_column_width=True)
            
            else:  # Manual
                st.write("Click on the image to select top-left and bottom-right corners for cropping.")
                st.write("Hover over the image to see pixel coordinates.")
                
                value = streamlit_image_coordinates(img_array, key="manual_crop")
                
                if value is not None:
                    st.write(f"Selected coordinate: {value}")
                
                if 'crop_coords' not in st.session_state:
                    st.session_state.crop_coords = []
                
                if value is not None and value not in st.session_state.crop_coords:
                    st.session_state.crop_coords.append(value)
                
                if len(st.session_state.crop_coords) == 2:
                    x1, y1 = st.session_state.crop_coords[0]
                    x2, y2 = st.session_state.crop_coords[1]
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    if st.button('Crop'):
                        cropped_img = crop_image(img_array, x1, y1, x2-x1, y2-y1)
                        st.image(cropped_img, caption='Cropped Image', use_column_width=True)
                        st.session_state.crop_coords = []
    
    else:
        st.write("Please upload an image to start processing.")

if __name__ == "__main__":
    main()