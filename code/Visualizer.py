import tkinter as tk
import cv2
import os
from HorizontalScrollableFrame import HSF
from PIL import Image
from PIL import ImageTk

test_dir = r"C:\Users\tyler\Desktop\phase2"
img_dir = r"C:\Users\tyler\Desktop\phase2test\cse515_dataset_2\Dataset2"
thumbnail_size = (160, 120)
ls_width = 1125
data_ls_height = 780
ftr_ls_height = 225


def create_thumbnail(img_id):
    # Load an image using OpenCV
    img_path = os.path.join(img_dir, img_id)
    # print('Loading image at path: %s' % img_path)
    cv_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    tn_img = cv2.resize(cv_img, thumbnail_size, interpolation=cv2.INTER_AREA)
    return tn_img


def visualize_matching_images(q_img, images_data, m, technique, fm, label):
    photos = []
    imgdata_to_visualize = []
    # Create a window
    window = tk.Tk()
    title_txt = "Visualization of Similar Images for %s with %s Feature Descriptors" % (technique, fm)
    if label != '':
        title_txt = title_txt + ' and label: ' + label
    window.title(title_txt)
    q_header = tk.Frame(window, relief=tk.RIDGE, borderwidth=2)
    q_lbl = tk.Label(q_header, text='Query Image')
    q_name = tk.Label(q_header, text='Query Image ID')
    """ rowspan set to 11 as that is the max count of rows for images 
        and data that will be stored for 5 or more similar images."""
    q_header.grid(row=0, column=0, columnspan=2, rowspan=11)
    q_lbl.grid(row=0, column=0)
    q_name.grid(row=0, column=1)
    # q_row = tk.Frame(window)
    q_cimg = create_thumbnail(q_img)
    q_canvas = tk.Canvas(q_header, width=thumbnail_size[0], height=thumbnail_size[1])
    q_photo = ImageTk.PhotoImage(image=Image.fromarray(q_cimg))
    # Add a PhotoImage to the Canvas
    q_canvas.create_image(0, 0, image=q_photo, anchor=tk.NW)
    # print('Giving label %s to last image loaded' % q_img)
    # print()
    q_id = tk.Label(q_header, text=q_img)
    q_canvas.grid(row=1, column=0)
    q_id.grid(row=1, column=1)

    img_label = tk.Label(window, text='Image ID')
    img_score = tk.Label(window, text='Matching Score')
    img_label.grid(row=0, column=2)
    img_score.grid(row=0, column=3)
    cur_row = 1
    img_col = 2
    score_col = 3
    count = 0
    for key, value in sorted(images_data.items(), key=lambda item: item[1]):
        if count < m:
            """ This will only happen if we have more than 5 similar images to visualize.
                In this case repeat the header frame above next 5 similar images. """
            if cur_row == 0:
                img_label = tk.Label(window, text='Image ID')
                img_score = tk.Label(window, text='Matching Score')
                img_label.grid(row=cur_row, column=img_col)
                img_score.grid(row=cur_row, column=score_col)
                cur_row += 1

            row = tk.Frame(window, relief=tk.RIDGE, borderwidth=2)

            tn_img = create_thumbnail(key)

            # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
            height, width, no_channels = tn_img.shape

            # Create a canvas that can fit the above image
            canvas = tk.Canvas(row, width=width, height=height)
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            photo = ImageTk.PhotoImage(image=Image.fromarray(tn_img))
            photos.append(photo)
            # Add a PhotoImage to the Canvas
            canvas.create_image(0, 0, image=photos[count], anchor=tk.NW)
            # print('Giving label %s to last image loaded' % key)
            # print()
            match_label = tk.Label(window, text=key)
            match_label.grid(row=cur_row, column=img_col, columnspan=2)
            """ After displaying the label of the image to be displayed up the current row value. """
            cur_row += 1
            label = tk.Label(row, text=str(value))
            row.grid(row=cur_row, column=img_col, columnspan=2)
            canvas.grid(row=cur_row, column=img_col)
            label.grid(row=cur_row, column=score_col)
            """ After adding the image thumbnail and score up the current row value. """
            cur_row += 1
            imgdata_to_visualize.append((canvas, label))
            """ Up the count after adding the image thumbnail and score. """
            count += 1
            """ If we have 5 images go ahead and reset current row to 0 so if more images occur we can display
                those images in the following column. Also increase the columns for images and scores by 2. """
            if count % 5 == 0:
                cur_row = 0
                img_col += 2
                score_col += 2
        else:
            break

    window.mainloop()


def visualize_data_ls(data_ls, technique, fm, label):
    """ Function to visualize the Data to Latent Semantics Matrix.
        Note k value is not needed here as the data_ls is a list of dataFrames of length k. """
    photos = []

    # Create a window
    window = tk.Tk()
    title_txt = "Visualization of Data-Latent Semantics for %s with %s Feature Descriptors" % (technique, fm)
    if label != '':
        title_txt = title_txt + ' and label: ' + label
    window.title(title_txt)

    frame = HSF(window, ls_width, data_ls_height)

    v_row = 0
    img_col = 0
    lbl_col = 1
    ls_count = 1
    p_count = 0
    """ ls_list is a list of tuples of (images, scores) """
    for ls_list in data_ls.values:
        ls_label = tk.Label(frame.scrollable_frame, text='Latent Semantic %s' % ls_count)
        ls_label.grid(row=v_row, column=img_col, columnspan=2)
        """ After displaying the latent semantic label up the current row value. """
        v_row += 1
        for img, score in ls_list:
            row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
            tn_img = create_thumbnail(img)
            # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
            height, width, no_channels = tn_img.shape
            # Create a canvas that can fit the above image
            canvas = tk.Canvas(row, width=width, height=height)
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            photo = ImageTk.PhotoImage(image=Image.fromarray(tn_img))
            photos.append(photo)
            canvas.create_image(0, 0, image=photos[p_count], anchor=tk.NW)
            """ Up the photo count by 1 so next image will be retrieved from the correct index """
            p_count += 1
            # print('Giving label %s to last image loaded' % img)
            # print()
            match_label = tk.Label(frame.scrollable_frame, text=img)
            match_label.grid(row=v_row, column=img_col, columnspan=2)
            """ After displaying the label of the image to be displayed up the current row value. """
            v_row += 1
            label = tk.Label(row, text=str(score))
            row.grid(row=v_row, column=img_col, columnspan=2)
            canvas.grid(row=v_row, column=img_col)
            label.grid(row=v_row, column=lbl_col)
            """ After adding the image thumbnail and score up the current row value. """
            v_row += 1

        """ After going through each list reset the row back to 0, and up the Latent Semantic count by 1
            Also increase the image column and label column each by 2. """
        v_row = 0
        ls_count += 1
        img_col += 2
        lbl_col += 2

    frame.pack(expand=True, fill='both')
    window.mainloop()


def visualize_feature_ls(feature_ls, technique, fm, label):
    """ Function to visualize the Feature to Latent Semantics Matrix.
        Note k value is not needed here as the data_ls is a list of dataFrames of length k. """
    # Create a window
    window = tk.Tk()
    title_txt = "Visualization of Feature-Latent Semantics for %s with %s Feature Descriptors" % (technique, fm)
    if label != '':
        title_txt = title_txt + ' and label: ' + label
    window.title(title_txt)

    frame = HSF(window, ls_width, ftr_ls_height)

    v_row = 0
    ftr_col = 0
    lbl_col = 1
    ls_count = 1
    """ ls_list is a list of tuples of (images, scores) """
    for ls_list in feature_ls.values:
        ls_label = tk.Label(frame.scrollable_frame, text='Latent Semantic %s' % ls_count)
        ls_label.grid(row=v_row, column=ftr_col, columnspan=2)
        """ After displaying the latent semantic label up the current row value. """
        v_row += 1
        row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
        feature_id = tk.Label(row, text="Feature Identifier", width=15)
        score_id = tk.Label(row, text="Feature Score", width=15)
        row.grid(row=v_row, column=ftr_col, columnspan=2)
        feature_id.grid(row=v_row, column=ftr_col)
        score_id.grid(row=v_row, column=lbl_col)
        """ After adding identifier labels up the current row value. """
        v_row += 1
        for feature, score in ls_list:
            data_row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
            feature_label = tk.Label(data_row, text=str(feature), width=14)
            score_label = tk.Label(data_row, text=str(score), width=16)
            data_row.grid(row=v_row, column=ftr_col, columnspan=2)
            feature_label.grid(row=v_row, column=ftr_col)
            score_label.grid(row=v_row, column=lbl_col)
            """ After adding the image thumbnail and score up the current row value. """
            v_row += 1

        """ After going through each list reset the row back to 0, and up the Latent Semantic count by 1
            Also increase the image column and label column each by 2. """
        v_row = 0
        ls_count += 1
        ftr_col += 2
        lbl_col += 2

    frame.pack(expand=True, fill='both')
    window.mainloop()
