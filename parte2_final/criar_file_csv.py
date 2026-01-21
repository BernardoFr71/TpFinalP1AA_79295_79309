import os
import cv2
import pandas as pd
from hand_landmark_extractor import HandLandmarkExtractor

def build_landmark_database(source_directory: str, destination_csv: str):
    """
    Percorre a diretoria de imagens, extrai coordenadas da mão e consolida num ficheiro CSV.
    """

    # Inicialização do processador MediaPipe
    # Mantive static_image_mode=True para maior precisão em fotos isoladas
    mp_processor = HandLandmarkExtractor(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        suppress_warnings=True
    )

    data_collection = []

    # Verificação de existência da pasta
    if not os.path.exists(source_directory):
        print(f"Erro: Diretoria não encontrada -> {source_directory}")
        return

    # Listar subpastas (Classes A, B, C...)
    # Filtra apenas o que for pasta para evitar erros com ficheiros soltos
    categories = sorted([d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))])

    for char_class in categories:
        class_path = os.path.join(source_directory, char_class)
        print(f"A extrair dados da classe: {char_class}...")

        # Loop pelos ficheiros dentro da pasta da classe
        for file_name in os.listdir(class_path):
            full_path = os.path.join(class_path, file_name)

            # Leitura da imagem
            raw_frame = cv2.imread(full_path)
            
            # Se a imagem estiver corrompida ou não for lida, salta
            if raw_frame is None:
                continue

            # Processamento: Extração de coordenadas
            results = mp_processor.process_image_landmarks(raw_frame)

            if results:
                # Conversão dos dados brutos para DataFrame
                # A função hands_data_to_dataframe já trata de aplanar os dados
                df_chunk = mp_processor.hands_data_to_dataframe(results)
                
                # Atribuição da etiqueta
                df_chunk['label'] = char_class
                
                data_collection.append(df_chunk)

    # Consolidação final do dataset
    if data_collection:
        full_dataset = pd.concat(data_collection, ignore_index=True)
        full_dataset.to_csv(destination_csv, index=False)
        print(f"Sucesso! Dataset guardado em: {destination_csv}")
        print(f"Total de registos gerados: {len(full_dataset)}")
    else:
        print("Aviso: Não foram detetados landmarks em nenhuma imagem.")

    # Libertar recursos do MediaPipe
    mp_processor.close()

if __name__ == "__main__":
    # Ajustado conforme o teu print (dataset/SignAlphaSet)
    ROOT_DIR = os.path.join("dataset", "SignAlphaSet")
    TARGET_FILE = "landmarks_dataset.csv"
    
    build_landmark_database(ROOT_DIR, TARGET_FILE)