from tkinter import *

from tkinter import filedialog

import imutils
from PIL import Image
from PIL import ImageTk

from face_emotion import *

import cv2


def select_image():
    actions = ['emotion']
    global panelA
    global prediction_text
    path = filedialog.askopenfilename()

    if len(path) > 0:
        image = cv2.imread(path)
        image = imutils.resize(image, width=720)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictions = classify_emotion(image, 'ssd', actions)

        image = Image.fromarray(predictions[0])
        prediction_text = predictions[1]
        image = ImageTk.PhotoImage(image)

        if panelA is None:
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="top", padx=10, pady=10)

        else:
            panelA.configure(image=image)
            panelA.image = image

        text = Label(window, text=prediction_text, bg='black', fg='white')
        text.pack(side="bottom", fill='both', expand='no', padx='10', pady='10')


window = Tk()
window.title('Face Emotion Recognizer')
window.configure(background='black')
panelA = None
prediction_text = None

btn = Button(window, text="Select an image", command=select_image, background='black', foreground='white')
btn.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")

if __name__ == '__main__':
    window.mainloop()
