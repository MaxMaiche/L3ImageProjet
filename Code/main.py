import os

import cv2 as cv

import validation_comparaison
from Code import tableau_detection_lignes, lignes_composantes_connexes
from lignes_rlsa import rlsa

validation_threshold = 0.9

pourcentages_tableau = []
pourcentages_lignes = []


def traitement(nom_image):
    base_image = cv.imread("../Ressources/VALIDATIONNEPASTOUCHER/" + nom_image)

    # •===• Obtention du tableau et des lignes •===•
    # Décommenter la ligne correspondant au traitement à tester

    # board, binary_board = tableau_composantes_connexes.get_board(img)
    board, binary_board, M = tableau_detection_lignes.get_board(nom_image)

    lines, binary_lines = lignes_composantes_connexes.get_lines(board, base_image, M)
    # board, binary_board = lignes_rlsa.get_board(img)

    # •===• Validation •===•

    # Obtention des images de validation
    nom_image = nom_image.split(".")[0]
    validation_board = cv.imread("../Validation/labeling/" + nom_image + "/board.png", cv.IMREAD_GRAYSCALE)
    validation_lines = cv.imread("../Validation/labeling/" + nom_image + "/lignes.png", cv.IMREAD_GRAYSCALE)

    # Calcul des scores
    score_board = validation_comparaison.compare(validation_board, binary_board)
    score_lines = validation_comparaison.compare(validation_lines, binary_lines)
    pourcentages_tableau.append(score_board)
    pourcentages_lignes.append(score_lines)

    # Affichage des scores
    print(f"""
    Image {nom_image}
    
    Score du tableau : {score_board} - Validation {"réussie" if score_board > validation_threshold else "échouée"}
    Score des lignes : {score_lines} - Validation {"réussie" if score_lines > validation_threshold else "échouée"}
    
    Validation finale : {"réussie" if score_board > validation_threshold and score_lines > validation_threshold else "échouée"}
    """)

    return 1 if score_board > validation_threshold else 0, 1 if score_lines > validation_threshold else 0


# TODO : déplacer du main
def getRlsa(image, horizontal, vertical):
    newimage = rlsa.rlsa(image, horizontal, vertical)
    return newimage


def do_all():
    folder_path = '../Ressources/VALIDATIONNEPASTOUCHER'
    allowed_extensions = '.jpg', '.png', '.jpeg'
    nb_validated_boards = 0
    nb_validated_lines = 0
    nb_validated_images = 0
    nb_images = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(allowed_extensions):
            if os.path.exists("../Validation/labeling/" + filename.split(".")[0]):
                board_ok, lines_ok = traitement(filename)
                nb_validated_boards += board_ok
                nb_validated_lines += lines_ok
                nb_validated_images += 1 if board_ok and lines_ok else 0
                nb_images += 1

    print(f"""
    •===• Résultats •===•
    
    Tableaux validés : {nb_validated_boards}/{nb_images} - {nb_validated_boards / nb_images * 100:.2f}%
    Lignes validées : {nb_validated_lines}/{nb_images} - {nb_validated_lines / nb_images * 100:.2f}%
    -> Images validées : {nb_validated_images}/{nb_images} - {nb_validated_images / nb_images * 100:.2f}%
    
    Liste des scores des tableaux : {pourcentages_tableau}
    Liste des scores des lignes : {pourcentages_lignes}
    """)


def do_one(nom_image):
    traitement(nom_image)


if __name__ == "__main__":
    do_all()
    # do_one("26.jpg")
