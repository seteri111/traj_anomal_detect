# Import system modules
import sys, string, os

# Export feature locations and attributes to an ASCII text file
 
# Import system modules
import arcpy
 
# Local variables...
inputdir = "D:/007/tasks/T-drive_Taxi_Trajectories/AIS_Data_gdb"
outputdir = "D:/007/tasks/T-drive_Taxi_Trajectories/AIS_Data_txt"
input_features = "Broadcast"
value_field = ['OBJECTID', 'BaseDateTime', 'SOG', 'COG', 'MMSI']

files = os.listdir(inputdir)
for f in files:
    if os.path.splitext(f)[1] == '.gdb':
        # Script arguments...
        inputfile = inputdir + os.sep + f
 
        # =============== file name process ======================
        basename = os.path.splitext(f)[0];
        export_ASCII = outputdir + os.sep + basename + ".txt";
 
        if os.path.exists(export_ASCII) == False:
            print inputfile
            # Process: Raster To Other Format (multiple)...
            try:
                # Set the current workspace (to avoid having to specify the full path to the feature classes each time)
                arcpy.env.workspace = inputfile
            
                # Process: Export Feature Attribute to ASCII...
                arcpy.ExportXYv_stats(input_features, value_field, 'COMMA', export_ASCII, "ADD_FIELD_NAMES")
            
            except:
                # If an error occurred when running the tool, print out the error message.
                print(arcpy.GetMessages())
 
            print export_ASCII