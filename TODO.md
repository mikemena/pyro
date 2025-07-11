[] In data_preprocessor -> def load_state -> add construct the path relative to the script's location so that the state file can be imported from any path

[] In train_model file -> def load_dataset -> load the preprocessor state to encode y if binary or categorical target

[] In train_model file -> def load_dataset -> change y tensor to this:
   y_tensor = torch.tensor(y, dtype=torch.float32)
