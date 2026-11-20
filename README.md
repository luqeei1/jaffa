# Marco - Hide and Seek Robot (HCR Team Jaffa)

This robot provides companionship and enrichment in entertainement for a wide audience. Primarily targeted at kids, especially those finding themselves stuck inside and alone for long periods of time. Marco is a companion robot with specialised sensors, skills and a human-first approach to improving the hide and seek game for all involved. 

Whether you are tired of arguing with your friends on who is next to seek or you are looking for someone to play with when all your friends cannot make it, Marco will be the ideal buddy to play with. 

## Integration demo

[`marco_search/README.md`](marco_search/README.md) is the run sheet for the integration demo.
Marco stops every few seconds, calls "Marco", looks with the RGB-D camera and listens for "Polo".
The belief map (`sensor_fusion/`) then picks a point in the SLAM map, and Nav2 drives there.
Everything is visible in Foxglove. A dry run without the robot works with `sim:=true`.
