string name                 # layer name, i.e. 'top1', 'bot1', 'bot2' etc.
string frame_id             # frame_id this belongs to, this should match the image set this is a part of
sensor_msgs/Image image		# Image data for top image
int64 count					# encoder count at time of acquisition start
int64 mm					# count->mm at time of acquisition start
