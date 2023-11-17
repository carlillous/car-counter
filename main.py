#apartado 2
import cv2
import numpy as np

# Creamos la instancia del extractor de fondo
bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
#bgSubtractor = cv2.createBackgroundSubtractorMOG2()

# Leemos el video
video = cv2.VideoCapture("trafico.mp4")
# Definimos un contador y el area de la región de interés.
area = np.array([[400,300], [655,300], [1280,580], [720,580]])
#area = np.array([[550, 400], [880, 400], [1800, 800],[1000, 800]]) #Este es el área si la imagen estuviese en dimensión original. [1100,500], [630,500] x1,y1 x2,y2
counter = 0
y1,y2 = 481, 488

while True:

    success, frame = video.read()
    if not success:
        break

    frame = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)  # 1920x1080 -> 1280x720
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Pasamos la imagen a escala de grises.

    #creamos una imagen auxiliar para pasar el extractor de fondo sobre la imagen extraida que nos interesa
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area], -1, (255), -1)
    roi = cv2.bitwise_and(gris, gris, mask=imAux)

    fgmask = bgSubtractor.apply(roi)

    #aplicamos operaciones morfológicas para mejorar la imagen e intentar eliminar el ruido.
    ret, thresh = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY) #pasamos la imagen a binario
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)   #aplicamos apertura: dilatación + erosión
    thresh = cv2.dilate(thresh, None, iterations=4)

    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            coords = (x, y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if  y1 < coords[1] < y2 :
                counter += 1
                print("Acabo de pasar! Total de coches: {}".format(counter))

    # Mostramos
    cv2.drawContours(frame, [area], -1, (255, 0, 255), 2)
    cv2.line(frame, (1070, y1), (615, y2), (0,0,255), 2)
    cv2.imshow("tresh",thresh)
    cv2.imshow("frame",frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()

