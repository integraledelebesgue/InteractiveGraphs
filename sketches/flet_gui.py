import flet as ft
from flet.matplotlib_chart import MatplotlibChart
import matplotlib.pyplot as plt
import networkx as nx
import time


def _main(page: ft.Page) -> None:
    t = ft.Text(value='Hello world', color='white')
    page.add(t)
    page.update()

    page.add(
        row := ft.Row(controls=[
            ft.Text('Text 1'),
            ft.Text('Text 2')
        ])
    )

    time.sleep(1)

    for i in range(1, 11):
        t.value = f'Step {i}...'
        row.controls.append(ft.Text(f'Appended text no. {i}'))
        row.controls.pop(0)
        page.update()
        time.sleep(0.5)

    def button_clicked(_):
        page.add(ft.Checkbox(label=new_task.value))

    new_task = ft.TextField(hint_text="Whats need to be done?", width=300)
    page.add(ft.Row([new_task, ft.ElevatedButton("Add", on_click=button_clicked)]))


def main(page: ft.Page) -> None:
    f, ax = plt.subplots()
    ax.axis('off')

    chart = MatplotlibChart(f, expand=True)

    target_rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=100,
        leading=ft.FloatingActionButton(icon=ft.icons.CREATE, text="New"),
        group_alignment=-0.9,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.icons.ADD_BOX_OUTLINED,
                selected_icon=ft.icons.ADD_BOX,
                label="Empty"
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.FILE_OPEN_OUTLINED,
                selected_icon=ft.icons.FILE_OPEN,
                label="From file"
            ),
        ],
        expand=True
    )

    edit_rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=100,
        leading=ft.FloatingActionButton(icon=ft.icons.CREATE, text="Edit"),
        group_alignment=-0.9,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.icons.ADD_BOX_OUTLINED,
                selected_icon=ft.icons.ADD_BOX,
                label="Add vertex"
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.SUBJECT_OUTLINED,
                selected_icon=ft.icons.OUTBOX_OUTLINED,
                label="Delete vertex"
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.FILE_OPEN_OUTLINED,
                selected_icon=ft.icons.FILE_OPEN,
                label="Add edge"
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.FILE_OPEN_OUTLINED,
                selected_icon=ft.icons.FILE_OPEN,
                label="Delete edge"
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.OUTBOX_OUTLINED,
                selected_icon=ft.icons.OUTBOX_OUTLINED,
                label="Change weight",
            )
        ],
        expand=True
    )

    left_rails_col = ft.Column(
        controls=[target_rail, edit_rail],
        alignment=ft.MainAxisAlignment.START,
        width=100
    )

    page.add(
        ft.Row(
            [
                left_rails_col,
                ft.VerticalDivider(width=1),
                ft.Column(
                    [chart],
                    alignment=ft.MainAxisAlignment.START,
                    expand=True
                )
            ],
            expand=True
        )
    )

    g = nx.Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(1, 4)

    nx.draw(g, ax=ax)
    page.update()


if __name__ == '__main__':
    ft.app(target=main)
