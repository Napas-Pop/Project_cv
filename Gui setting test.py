
    
import pyto_ui as ui

view = ui.View()
view.background_color = ui.COLOR_SYSTEM_BACKGROUND

label = ui.Label()
label.text_alignment = ui.TEXT_ALIGNMENT_CENTER
label.size = (view.width, 50)
label.flex = [ui.FLEXIBLE_WIDTH]
view.add_subview(label)

def did_end_editing(sender):
    if sender.text != "":
        u = sender.text
        print(u)
        view.close()
    return o == u
    

def did_change_text(sender):
    if sender.text == "":
        label.text = ""
    else:
        label.text = "marker size"+sender.text

text_field = ui.TextField(placeholder="What the size of marker?")
text_field.become_first_responder()
text_field.action = did_change_text
text_field.did_end_editing = did_end_editing
text_field.width = 200
text_field.center = (view.width/2, view.height/2)
text_field.flex = [
    ui.FLEXIBLE_BOTTOM_MARGIN,
    ui.FLEXIBLE_TOP_MARGIN,
    ui.FLEXIBLE_LEFT_MARGIN,
    ui.FLEXIBLE_RIGHT_MARGIN
]
view.add_subview(text_field)

ui.show_view(view, ui.PRESENTATION_MODE_SHEET)

print('test',o)



