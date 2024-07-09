import streamlit as st
import numpy as np
import cv2

def image_illumination(f, x0, y0, sigma):
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
            ('Image Illumination', 'Image Quantization', 'Image Downsampling', 'Image Cropping')
        )
        
        if processing_option == 'Image Illumination':
            nr, nc = img_array.shape[:2]
            x0 = st.slider('X0', 0, nr, nr // 2)
            y0 = st.slider('Y0', 0, nc, nc // 2)
            sigma = st.slider('Sigma', 1, 500, 200)
            processed_img = image_illumination(img_array, x0, y0, sigma)
            st.image(processed_img, caption='Processed Image', use_column_width=True)
        
        elif processing_option == 'Image Quantization':
            bits = st.slider('Bits', 1, 8, 4)  
            processed_img = image_quantization(img_array, bits)
            st.image(processed_img, caption='Processed Image', use_column_width=True)
        
        elif processing_option == 'Image Downsampling':
            sampling_rate = st.slider('Sampling Rate', 2, 10, 2)
            
            processed_img = image_downsampling(img_array, sampling_rate)
            st.image(processed_img, caption='Processed Image', use_column_width=True)
        
        elif processing_option == 'Image Cropping':
            h, w = img_array.shape[:2]
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.number_input("X1", 0, w-1, 0)
                y1 = st.number_input("Y1", 0, h-1, 0)
            with col2:
                x2 = st.number_input("X2", x1+1, w, w)
                y2 = st.number_input("Y2", y1+1, h, h)
            
            cropped_img = crop_image(img_array, x1, y1, x2-x1, y2-y1)
            st.image(cropped_img, caption='Processed Image', use_column_width=True)
            

    
    else:
        st.write("Please upload an image to start processing.")

if __name__ == "__main__":
    main()