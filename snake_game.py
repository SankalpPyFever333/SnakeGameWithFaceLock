import random
import pygame
import pandas
import numpy
import PIL.Image
import os
import cv2
import csv


########### face lock code ##############################


########### showing texts messages for registration and play game#############

def newRegistration():
    gameWindow.fill(blue)
    exit_game=False
    while not exit_game:

        screen_text("Press N for new gamer",white,300,300)
        screen_text("Press space for old gamer",red,400,400)
        for event in pygame.event.get():
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    TakeImages()
                if event.key == pygame.K_SPACE:
                    TrackImages()
            if event.type== pygame.QUIT:
                exit_game=True
        pygame.display.update()
        clock.tick(fps)
    pygame.quit()
    quit()
    
####################################


def ErrorMessage():
    exit_game= False

    while not exit_game:
        gameWindow.fill(white)
        screen_text("Some files are  missing",blue,300,300)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                exit_game=True
        pygame.display.update()
        clock.tick(fps)
    pygame.quit()
    quit()


#######################################


def checkFileExistence(path):
    dir= os.path.dirname(path)
    # print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)


###############################################

def TrackImages():
    callWelcome= False
    harcascade = "Face.xml" 
    checkFileExistence("PlayerDetails/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    exist3 = os.path.isfile("TrainingImageFolder/ImageTrainer.yml")
    if exist3:
        recognizer.read("TrainingImageFolder/ImageTrainer.yml")
    else:
        ErrorMessage()
    facecascade= cv2.CascadeClassifier(harcascade)
    cam= cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    exist1 = os.path.isfile("PlayerDetails/playerInfo.csv")
    if exist1:
        df = pandas.read_csv("PlayerDetails/playerInfo.csv")
    else:
        pass
    while True:
        ret,image= cam.read()
        cv2.flip(image,0)
        pygame.surfarray.make_surface(image)
        pygame.display.update()
        gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces= facecascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (random.randint(34, 200),random.randint(56, 228), random.randint(90, 234)), 3)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf<50:
                # aa= df.loc[df["SNO"]==serial]["NAME"].values
                bb="Detected"
                callWelcome=True
                
            else:
                bb="Unknown"
        cv2.putText(image, str(bb), (x, y+h), font, 2, (random.randint(0,240), random.randint(0, 140), random.randint(120, 245)),2)
        cv2.imshow("Recognizing face",image)
        if cv2.waitKey(1) == ord('q'):
            break
        # cv2.waitKey(10000)
    cam.release()
    cv2.destroyAllWindows() 
    if callWelcome:
        welcome()
    else:
        UnknownPlayer()

#########################################


def UnknownPlayer():
    exit_game = False

    while not exit_game:
        gameWindow.fill((212,243,180))
        screen_text("Unknown player! please Register Yourself", black, 300, 300)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_game = True
        pygame.display.update()
        clock.tick(fps)
    gameWindow.fill(blue)




################################################

def TakeImages():
    column= ['SNO','NAME']
    name=PlayerDetail()
    # print(name)
    # gameWindow.fill(white)
    checkFileExistence('PlayerDetails/')
    checkFileExistence('TrainingImages/')
    serial=0
    exists=os.path.isfile("PlayerDetails/playerInfo.csv")
    if exists:
        with open("PlayerDetails/playerInfo.csv","a+") as playerFile:
            reader= csv.reader(playerFile)
            for row in reader:
                serial+=1
            # print("before dividing serial",serial)    
            serial= (serial//2)
            # print("After dividing serial",serial)    
        playerFile.close()
    else:
        with open("PlayerDetails/playerInfo.csv", "a+") as playerFile:
            writer= csv.writer(playerFile)
            writer.writerow(column)
            serial=1
        playerFile.close()
    cam= cv2.VideoCapture(0)
    harcascade="Face.xml"
    detector= cv2.CascadeClassifier(harcascade)
    sampleNum=0
    while(True):
        success,img= cam.read()
        cv2.flip(img,0)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        pygame.surfarray.make_surface(img) #displaying the image on the pygame surface.
        pygame.display.update()
        faces= detector.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (random.randint(34, 200),random.randint(56, 228), random.randint(90, 234)), 3)
            sampleNum+=1
            # imagePath = "TrainingImages\\"+name+"." +str(serial) + "."+str(sampleNum)+".jpg"
            # print(os.getcwd())
            # newPath=os.chdir("TrainingImages\\")
            cv2.imwrite("TrainingImages\\" + str(sampleNum)+ ".jpg", gray[y:y + h, x:x + w])
            # print("TrainingImages\\" + name + "." + str(serial) +"." + str(sampleNum) + ".jpg") This path doesn't work becoz it may be due to the exceede path length.
            # print("TrainingImages\\" + str(sampleNum) + ".jpg")
            
            # gray[y:y + h, x:x + w]: we are saving that grayscale image those dimension.
            cv2.imshow("Taking Image",img)
        # if cv2.waitKey(0)==ord('q'):
        #     break
        if sampleNum>105:
            break
        cv2.waitKey(1)
    cam.release()
    cv2.destroyAllWindows()
    row=[serial,name]
    # print(row)
    with open("PlayerDetails/playerInfo.csv","a") as playerFile:
        writer= csv.writer(playerFile)
        writer.writerow(row)
    playerFile.close()
    # gameWindow.fill(skyBlue)
    TrainAlgorithm()

##################################################################
def GetImagesForTraining(path):
    imagePath=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    Ids=[]
    for ipath in imagePath:
        pilImage= PIL.Image.open(ipath).convert("L")
        imageNp=numpy.array(pilImage,'uint8')
        ID= int(os.path.split(ipath)[1].split(".")[0])
        # print(Ids)
        faces.append(imageNp)
        Ids.append(ID)
    return faces,Ids



#######################################################################

def TrainAlgorithm():
    harcascade="Face.xml"
    checkFileExistence("TrainingImageFolder/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector= cv2.CascadeClassifier(harcascade)
    faces,ID = GetImagesForTraining("TrainingImages")
    try:
        recognizer.train(faces,numpy.array(ID))
    except:
        Noplayer()
    recognizer.save("TrainingImageFolder/ImageTrainer.yml")
    profileSaved()
    
################################################


def Noplayer():
    exit_game = False

    while not exit_game:
        gameWindow.fill(white)
        screen_text("No Player found", blue, 300, 300)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_game = True
        pygame.display.update()
        clock.tick(fps)
    pygame.quit()
    quit()


################################################

def profileSaved():
    exit_game = False

    while not exit_game:
        gameWindow.fill(white)
        screen_text("Profile saved successfully", black, 300, 300)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_game = True
        pygame.display.update()
        clock.tick(fps)
    gameWindow.fill(blue)




# ################## Entering player detail#############
def PlayerDetail():
    exit_game= False
    # gameWindow.fill(skyBlue)
    playername=""
    # pygame.draw.rect(gameWindow,white,[440,300,200,40])
    while not exit_game:
        # screen_text("Player Name:",white,300,300)
        for event in pygame.event.get():
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_BACKSPACE:
                    playername=playername[:-1]
                    # screen_text(playername,black,450,310)
                else:
                    if event.key!=pygame.K_RETURN:
                        playername= playername+event.unicode
                    # screen_text(playername,black,450,310)
                    
                    
                if event.key==pygame.K_RETURN:
                    return playername
                    # return playername[:len(playername)-1]
                    # Here we are doing one more time slicing to remove the last element bcoz after typing name we press enter key(\r) and that gets concated to playername.
                if event.type==pygame.QUIT:
                    exit_game=True
        gameWindow.fill(skyBlue)
        screen_text("Player Name:", white, 300, 300)
        pygame.draw.rect(gameWindow, white, [440, 300, 200, 40])
        screen_text(playername, black, 450, 310)
        pygame.display.update()
        clock.tick(fps)



############### SNAKE GAME CODE###################################


pygame.mixer.init() # mixer gets initialise
# we are here only loading the  music file
pygame.init() # initialising all the modules of pygame library.


################################################


# Opening highscore.txt file in read mode:
# with open("highscore.txt","r") as f:
#     highscr= f.read() #Reading highscore.txt file
#     # So, when we read a file , its value come as a string.
# IF YOU READ THAT FILE HERE, IT WILL RAISE REFERENCE ERROR.


# putting score on screen:
def screen_text(text, color, x, y):
    # We want to put text ,of color, and where to put(x,y)
    screen_text = font.render(text, True, color)
    # anti-aliasing (search it). It is the second argument provided to render().antialias
    gameWindow.blit(screen_text, [x, y])
    # blit() is used to update screen
    # using blit(), we put our one surface on the main window(gameWindow). Here, we are putting the screen_text on the gameWindow surface.
    # refer to pygame documentation


def plot_snake(gameWindow, color, snk_list, s_size):
    for x, y in snk_list:
        pygame.draw.rect(gameWindow, color, [x, y, s_size, s_size])
        # x,y which we have given here are the coordinates of the top left coordinates of this rectangle.
        
        # drawing red dots on rectangle
        pygame.draw.circle(gameWindow, (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255)), (x+2, y+3), 3, 5)


# pygame.draw.rect(gameWindow, black, [food_x, food_y, food_size, food_size])
# When you place food code here, it will not appear bcoz this food will appear first then we are filling the window with white color that's why it will disappear. So place this code after filling the color.

# def menuscreen():
#     exit_game=False
#     while not exit_game:
#         gameWindow.fill((26,67,134))
#         screen_text("Game Type", blue, 70, 500)
#         screen_text("level", blue, 70, 500)
#         for event in pygame.event.get():
#             if event.type==pygame.quit():
#                 exit_game=True
#         pygame.display.update()
#         clock.tick(fps)

#########################################################

def welcome():
    exit_game=False
    width_height=(250,490)
    iconimg = pygame.image.load("newSnakeFile.webp")
    pygame.display.set_icon(iconimg)
    while not exit_game:
        gameWindow.fill((23,240,40))
        screen_text("Welcome to SnakeGame",black,200,300)
        # screen_text("press Enter to play!", white,234,390)
        # So, here when this function runs, a welcome screen will appear but our game will not start bcoz we had not called the gameloop() function, so here we write when you press enter, gameloop() function start running.
        pygame.draw.rect(gameWindow,red,[250,490,40,25])
        screen_text("Play",black,250,490)
        a= pygame.mouse.get_pos()
        mytuple= tuple(map(lambda i,j:abs(i-j),a,width_height))
        if mytuple[0]<40 and mytuple[1]<25:
            for event in pygame.event.get():
                # The MOUSEBUTTONDOWN event occurs once when you click the mouse button
                # MOUSEBUTTONDOWN event occurs when you click the mouse button, either the left or right, regardless of how much time you hold it after clicking.
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pygame.draw.rect(gameWindow, yellow, [250, 490, 40, 25])
                    screen_text("Play", green, 250, 490)
                    GameLoop()
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                exit_game=True
            # if event.type==pygame.KEYDOWN:
            #     if event.key==pygame.K_RETURN:
            #         GameLoop()
        pygame.display.update()
        clock.tick(fps)
    pygame.quit()
    quit()


##############################################################

# Game loop
def GameLoop():
    exit_game = False
    game_over = False
    # initially position in x coordianate
    snake_x = random.randint(35, screen_width-50)
    # initital position in y coordinate
    snake_y = random.randint(60, screen_height-60)
    s_size = 10
    init_vel = 8
    vel_x = 0
    vel_y = 0
    # s_list = []
    snk_list = []
    snake_length = 1
    score = 0
    bigFoodCount=0
    with open("highscore.txt", "r") as f:
        highscr = f.read()  # Reading highscore.txt file
    # So, when we read a file , its value come as a string.
    # generates an integer between 0 to screen_width
    food_x = random.randint(50, screen_width//2)

    # generates an integer between 0 to screen_heigth
    food_y = random.randint(50, screen_height//2)
    gameWindow.fill((random.randint(0, 255), random.randint(
        0, 255), random.randint(0, 255)))
    bigFood_x = random.randint(20, screen_width/2)
    bigFood_y = random.randint(20, screen_height/2)
    iconimg = pygame.image.load("newSnakeFile.webp")
    pygame.display.set_icon(iconimg)
    # for playing two or more music simultaneously, we have to make channel.
    # pygame.mixer.music.load("ManjhaTera.mp3")
    # pygame.mixer.music.play()
    pygame.mixer.Channel(1).play(pygame.mixer.Sound("ManjhaTera.mp3"),loops=-1)
    # loops=-1 means sound will play infinetely.
    pygame.mixer.music.set_volume(0.4)
    while not exit_game:
        # mainpos=(70,500)
        # a=pygame.mouse.get_pos()
        
        # maincurpos=(tuple(map(lambda i,j:abs(i-j),mainpos,a)))
        # pygame.mouse.set_visible(False)
        # curosr is not visible
        # gameWindow.blit(iconimg, (0, 0))
        if game_over:
            # here, we want to write our hiscore in the file:
            # so we open thw same file in write mode:
            with open("highscore.txt","w") as f:
                f.write(str(highscr))
                # We write content in file in string only, so we have converted it into string.
            gameWindow.fill(white)
            screen_text("Game Over!Press Enter to Play", red, 70, 300)
            pygame.draw.rect(gameWindow,red,[70,500,107,20])
            screen_text("Main Menu",blue,70,500)
            for event in pygame.event.get():
                # print(event)
                if event.type == pygame.QUIT:  # if we quit the window,game close
                    exit_game = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:  # handling the enter key
                        welcome() # Here we are running welcome function bcoz we want that when game will over this welcome screen appear again and then our gameloop() will start.And game will start again.
            # if maincurpos[0] <= 107 and maincurpos[1] <= 20:
            #     for event in pygame.event.get():
            #         if event.type == pygame.MOUSEBUTTONDOWN:
            #             menuscreen()
        else:
                
                for event in pygame.event.get():
                    # print(event)
                    if event.type == pygame.QUIT:
                        exit_game = True
                    # if we press any arrow key then :
                    # logic for pausing the game: check here whtether the cursor is on pause or not, if yes then do not add velocity to it otherwise add it.
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RIGHT:
                            vel_x = init_vel
                            vel_y = 0
                            
                        if event.key == pygame.K_LEFT:
                            vel_x = -init_vel
                            vel_y = 0
                            
                        if event.key == pygame.K_UP:
                            vel_y = -init_vel  # -ve Y axis is in up direction
                            vel_x = 0
                            
                        if event.key == pygame.K_DOWN:
                            vel_y = init_vel
                            vel_x = 0
                            
                rectpos = (300, 5)
                curpos = pygame.mouse.get_pos()
                respos = tuple(map(lambda x, y: abs(x-y), curpos, rectpos))
                if respos[0] <= 60 and respos[1] <= 20:
                    for event in pygame.event.get():
                        if event.type == pygame.MOUSEBUTTONUP:
                            snake_x=snake_x
                            snake_y=snake_y
                else:
                    snake_x = snake_x+vel_x
                    snake_y = snake_y+vel_y

                if abs(snake_x-food_x) < 8 and abs(snake_y-food_y) < 8:
                    score = score+1
                    pygame.mixer.music.load('Paper Ripping.mp3')
                    pygame.mixer.music.play()
                    snake_length = snake_length+8
                    bigFoodCount+=1
                    

                    # Now we change the position the position of food by random module.
                    food_x = random.randint(35, screen_width-50)
                    food_y = random.randint(60, screen_height-60)
                if score > int(highscr): #We are converting high score to int bcoz when we read a file , it come as a string.
                    highscr=score
                # logic for bigger food:
                if bigFoodCount>=5:
                    pygame.mixer.Channel(2).play(pygame.mixer.Sound("soundEffect3.wav"))
                    pygame.draw.circle(gameWindow, (random.randint(0, 34), random.randint(
                        0, 125), 87), (bigFood_x,bigFood_y), 10, 7)
                    pygame.display.update()
                    if abs(snake_x-bigFood_x) <= 15 and abs(snake_y-bigFood_y) <= 15:
                        score=score+5
                        pygame.mixer.Channel(2).play(pygame.mixer.Sound("soundEffect2.wav"))
                        bigFoodCount=0
                        bigFood_x = random.randint(25, screen_width-40)
                        bigFood_y = random.randint(60, screen_height-60)

                # here you will not get this white color bcoz when you change anything in dispaly , you have to run pygame.display.update() function.
                # gameWindow.fill((random.randint(0,255),random.randint(0,255),random.randint(0,255)))

                gameWindow.fill((123,234,160))
                # putting score on the screen. We have to pass score as a string.
                screen_text("score:"+str(score), red, 5, 5)
                screen_text("  High Score:"+str(highscr), blue, 90, 5)
                # We are calling this function here bcoz after filling white color, then we want to put score on window. If we call it before filling then score will not appear.
                pygame.draw.rect(gameWindow, black, [300, 5, 60, 20])
                screen_text("Pause", blue,300, 5)
                #pygame.time.delay()
                # pause the program for an amount of time

                # creating side walls

                pygame.draw.rect(gameWindow, (0, 0, 0), [10, 30, screen_width-20, 9])
                pygame.draw.rect(gameWindow, (0, 0, 0), [screen_width-20, 30, 9, screen_height-48])
                pygame.draw.rect(gameWindow, (0, 0, 0), [10, screen_height-25, screen_width-20, 9])
                pygame.draw.rect(gameWindow, (0, 0, 0), [10, 30, 9, screen_height-48])

                # walls in between:
                pygame.draw.rect(gameWindow, (0, 0, 0), [200, 260, 400, 9])
                pygame.draw.rect(gameWindow, (0, 0, 0), [200, 400, 400, 9])




                head = []
                # here, we are appending the position of x,y only when the mouse is not on the puase button.
                rectpos = (300, 5)
                curpos = pygame.mouse.get_pos()
                respos = tuple(map(lambda x, y: abs(x-y), curpos, rectpos))
                if respos[0] <= 60 and respos[1] <= 20:
                    pass
                else:
                    head.append(snake_x)
                    head.append(snake_y)
                    snk_list.append(head)                                        
                if len(snk_list) > snake_length:
                    del snk_list[0]
                    # this is used to remove the first element of  list when snk_list is greater than snake_length list.
                    # So, when you eat food, that will be appended in the snk_list as head.and then snk_list will become greater than snake_length.
                    # Snake length is increaese by 10 coordinates every time when snake eats food.
                if head in snk_list[:-1]: #here we are excluding the last element of list by doing list slicing, if we do not d0 this then our game will over as it start, becoz initially , the first element is the last element of list. So, this condition gets true and our game  will over
                    game_over=True
                    # pygame.mixer.music.load("Gunfire And Voices.mp3")
                    pygame.mixer.Channel(2).play(pygame.mixer.Sound("Gunfire And Voices.mp3"))
                    pygame.time.delay(1500)
                    pygame.mixer.music.stop()

                pygame.draw.rect(gameWindow, (154,random.randint(
                    0, 153), random.randint(80, 190)), [food_x, food_y, food_size, food_size])
                # pygame.draw.rect(gameWindow,red,[snake_x,snake_y,s_size,s_size])
                if snake_x <= 18 or snake_x >= screen_width-20 or snake_y >= screen_height-28 or snake_y <= 32:
                    game_over = True
                    pygame.mixer.music.load("Gunfire And Voices.mp3")
                    pygame.mixer.music.play()
                    pygame.time.delay(1400)
                    pygame.mixer.music.stop()
                
                # checking whether snake head hit by walls in between:

                # checking for down wall:
                
                if snake_x>=200 and snake_x<=600:
                    
                    if abs(snake_y-260)<=9:
                        
                        game_over = True
                        pygame.mixer.music.load("Gunfire And Voices.mp3")
                        pygame.mixer.music.play()
                        pygame.time.delay(1400)
                        pygame.mixer.music.stop()
                if snake_x>=200 and snake_x<=600:
                    
                    if abs(snake_y-400)<=9:
                        
                        game_over = True
                        pygame.mixer.music.load("Gunfire And Voices.mp3")
                        pygame.mixer.music.play()
                        pygame.time.delay(1400)
                        pygame.mixer.music.stop()

                
                
                



                plot_snake(gameWindow, black, snk_list, s_size)
                # rect(surface(where you want to be place), color, (x and y corrdinate), width=0, border_radius=0, border_top_left_radius=-1, border_top_right_radius=-1, border_bottom_left_radius=-1, border_bottom_right_radius=-1)
                
        pygame.display.update()
        clock.tick(fps) 
            # This tells that how many times the while loop is going to run per second, here it will run 50 times per second.

            # in game loop, keep less code as much as possible.
    pygame.quit()
    quit()
    # sys.exit()

############# variables###########

playername = ""
fps = 25
# frame per second is : the number of images consecutively displayed each second -- and is a common metric used in video capture and playback when discussing video quality.The human brain can only process about 10 to 12 FPS. Frame rates faster than this are perceived to be in motion. The greater the FPS, the smoother the video motion appears. Full-motion video is usually 24 FPS or greater.


# defining the colors:
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
green=(10,250,1)
yellow=(255,255,80)
lightBlue = (32, 32, 134)
grey = (100, 98, 63, 0.795)
skyBlue = (14, 238, 227, 0.808)
screen_width = 800
screen_height = 600
food_size = 8
bigFoodSize = 14

#########################################

# creating game window
gameWindow = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("SnakeGame")
pygame.display.update()  # It allows only a portion of the screen to updated, instead of the entire area. If no argument is passed it updates the entire Surface area
# defining clock
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)
# font=pygame.font.SysFont(font type,font size)
# None is given when we want to put the system default font.


newRegistration()

# Add a song in background : Give the button to setting volume.
# made walls
# give the option to change the background song randomly.
# add a udti chidiya effect when snake gets hit.
#  Add a time limit upto which the bigFood is present in the screen.
# Level change after sometime.
# if snake gets inside fom one side immediately appear its oppsite side.

