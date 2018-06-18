from sklearn.externals import joblib
import os

def loading_model(file_name):
    # model loading
    current_dir = os.getcwd()
    model_file = os.path.join(current_dir, file_name)
    model = joblib.load(model_file)
    
    return model

def char_recog(model, characters, column_list):
    classification_result = []
    for each_character in characters:
        # convert 1D array
        each_character = each_character.reshape(1,-1);
        result = model.predict(each_character)
        classification_result.append(result)
        
    plate_string = ''
    for each_predict in classification_result:
        plate_string += each_predict[0]
        
        # its possible that char are wrongly arranged 
        # the column_list will be
        # used to sort the letters in the right order

    column_list_copy = column_list[:]
    column_list.sort()
    correct_plate = ''
    for each in column_list:
        correct_plate += plate_string[column_list_copy.index(each)]
        
    return correct_plate
