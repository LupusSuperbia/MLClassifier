import numpy as np 
import matplotlib.pyplot as plt 
import streamlit as st 
import os
from PIL import Image

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def main(): 
    st.title('Cifar10 Web Classifier')
    st.write('Suba cualquier imagen que creas que sea en una de las clases y comprueba si la perdicción es correcta')
    
    file = st.file_uploader('Porfavor subo la imagen', type=['jpg', 'png', 'webp'])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image) / 255

        img_array = img_array.reshape((1, 32, 32, 3))

        # Obtiene la ruta del directorio actual
        current_directory = os.getcwd()

        # Nombre del archivo del modelo
        model_filename = 'Current/cifar10_model.h5'

        # Combina la ruta actual con el nombre del archivo del modelo
        model_path = os.path.join(current_directory, model_filename)

        if os.path.exists(model_path):
        # Carga el modelo desde el archivo cifar10_model.h5
            model = tf.keras.models.load_model(model_path)

            predictions = model.predict(img_array)
            cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck', 'dog']

            fig, ax = plt.subplots()
            y_pos = np.arange(len(cifar10_classes))
            ax.barh(y_pos, predictions[0], align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cifar10_classes)
            ax.invert_yaxis()
            ax.set_xlabel('Probability')
            ax.set_title('CIFAR10 Predictions')
            st.pyplot(fig=fig)
            pass
        else:
            st.text(f"No se encontró el archivo {model_filename} en el directorio {current_directory}. Verifica la ruta y el nombre del archivo.")

    else : 
        st.text('No has subido ninguna imagen')


if __name__ == '__main__':
    main()