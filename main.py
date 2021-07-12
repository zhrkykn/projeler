import cv2
import numpy as np
import dlib

webcam = True

cap = cv2.VideoCapture(0) # kamera bul
detector = dlib.get_frontal_face_detector() #yüz bulma
predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #yüzdeki bölgeleri noktalayan dosyamız
def empty(a):# trackbar ın hangi değerde olduğunu geri dödüren fonsiyon boş geçtik
    pass
cv2.namedWindow("BGR")
cv2.resizeWindow("BGR",640,240)
cv2.createTrackbar("Mavi",'BGR',0,255,empty)
cv2.createTrackbar("Yesil",'BGR',0,255,empty)
cv2.createTrackbar("Kirmizi",'BGR',0,255,empty)q


def createBox(img,points,scale=5,masked=False,cropped = True): # bir fonksiyon oluşturarak istediğimiz bölgeyi kırpacağız
    if masked:
       mask = np.zeros_like(img) # dudağı kırptık
       mask = cv2.fillPoly(mask,[points],(255,255,255)) #beyaz olarak maske çıkarttık
       img = cv2.bitwise_and(img,mask) #iki resmi birleştirdik
       #cv2.imshow('mask', img) #beyazı silip orjinali kırptık
    if cropped:
       bbox =cv2.boundingRect(points)
       x,y,w,h = bbox #çerçeveledik
       imgCrop = img[y:y+h,x:x+w]
       imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)# kırpacağız yer küçük olacağı için yeniden boyutlandırdık
       return imgCrop
    else:
       return mask
while True:#görüntüyü döngüye koyduk ki sürekli ekranda kalsın
    if webcam: success , img = cap.read()
    else:img = cv2.imread('1.jpg') #resmi içe aktardık.
    img = cv2.resize(img,(0,0),None,1,1) #resmi küçülttük
    imgOriginal = img.copy() #eşitledik
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #gri tonlamaya dönüştürdük
    faces = detector(imgGray) #yüzler diye değişkene atadık

    for face in faces:
        x1,y1 = face.left(),face.top() #yüzdeki sol nokta
        x2,y2 = face.right(),face.bottom() #sınırlayıcı kutu oluşturuyoruz
        #imgOriginal = cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2) #yüz köşeleri rengi ve kalınlığı
        landmarks = predictor(imgGray,face)
        myPoints = []

        for n in range(68): #yüzdeki 68 noktayıda bulmak için
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])
            #cv2.circle(imgOriginal,(x,y),3,(50,50,255),cv2.FILLED) #yüzdeki noktaları çizip yarıçap renk verdik
            #cv2.putText(imgOriginal,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,0,255),1) #hangi numaralar nereyi gösteriyor tüm özellikleri
        myPoints = np.array(myPoints)

        imgLips = createBox(img,myPoints[48:61],2,masked=True,cropped=False) #dudakları bulduk ölçekledik
        imgColorLips = np.zeros_like(imgLips)
        b = cv2.getTrackbarPos('Mavi','BGR')
        g = cv2.getTrackbarPos('Yesil','BGR')
        r = cv2.getTrackbarPos('Kirmizi','BGR')

        imgColorLips[:] =b,g,r

        imgColorLips = cv2.bitwise_and(imgLips,imgColorLips) #resimleri birleştirip dudağı mora boyadık

        imgColorLips = cv2.GaussianBlur(imgColorLips,(7,7),10)#bulanıklaştırdık yoksa yapay duruyor


        imgColorLips = cv2.addWeighted(imgOriginal, 1, imgColorLips, 0.4, 0)

        cv2.imshow('BGR',imgColorLips)


        cv2.imshow('dudaklar',imgLips) #dudakları kırptık

        print(myPoints) # bütün x ve y noktalarını yazdırdık
    cv2.imshow("orjinal",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break