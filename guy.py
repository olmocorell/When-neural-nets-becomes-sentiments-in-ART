from tkinter import *
from PIL import ImageTk
import PIL.Image
from flujo.ejecuciontkinter import *

def main():
    global window
    HEIGHT = 600
    WIDTH = 600
    window = Tk()
    window.title("WHEN NEURAL NETS BECOMES EMOTIONS IN ART")
    canvas = Canvas(window, height=HEIGHT, width=WIDTH)
    canvas.pack()


    # background
    image = PIL.Image.open("sentimientos/fondo.jpg")
    background_image = ImageTk.PhotoImage(image)
    background_label = Label(window, image=background_image)
    background_label.place(relwidth=1, relheight=1)


    #Para el boton1
    frame = Frame(window, bg='white', bd=5)
    frame.place(relx=0.302, rely=0.16, relwidth=0.375,
                relheight=0.1, anchor='n')
    button = Button(frame,text="Empieza a crear")
    button.place(relx=0, relheight=1, relwidth=1)
    button.configure(relief='groove',command= lambda: primeraParte())

    #Para el boton
    frame = Frame(window, bg='white', bd=5)
    frame.place(relx=0.302, rely=0.25, relwidth=0.375,
                relheight=0.1, anchor='n')
    button = Button(frame,text="Segunda parte",)
    button.place(relx=0, relheight=1, relwidth=1)
    button.configure(relief='groove',command= lambda: segundaParte())



    window.mainloop()


if __name__ == "__main__":
        main()