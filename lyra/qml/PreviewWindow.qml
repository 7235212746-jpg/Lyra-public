import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window

Window {
    id: viewer
    width: 1080
    height: 760
    minimumWidth: 640
    minimumHeight: 480
    visible: false
    title: "Lyra 预览"
    color: "#EEF3F2"

    property url source
    property real zoom: 1.0

    function open() {
        zoom = 1.0
        show()
        raise()
        requestActivate()
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 48
            color: "#FAFCFC"
            border.color: "#D8E2E0"
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                ToolButton {
                    text: "−"
                    ToolTip.visible: hovered
                    ToolTip.text: "缩小"
                    onClicked: viewer.zoom = Math.max(0.25, viewer.zoom - 0.15)
                }
                Slider {
                    from: 0.25
                    to: 4.0
                    value: viewer.zoom
                    onMoved: viewer.zoom = value
                    Layout.preferredWidth: 220
                }
                ToolButton {
                    text: "+"
                    ToolTip.visible: hovered
                    ToolTip.text: "放大"
                    onClicked: viewer.zoom = Math.min(4.0, viewer.zoom + 0.15)
                }
                ToolButton {
                    text: "1:1"
                    ToolTip.visible: hovered
                    ToolTip.text: "实际大小"
                    onClicked: viewer.zoom = 1.0
                }
                Item { Layout.fillWidth: true }
                Label { text: Math.round(viewer.zoom * 100) + "%" }
            }
        }
        Flickable {
            id: viewport
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            contentWidth: Math.max(width, image.implicitWidth * viewer.zoom + 48)
            contentHeight: Math.max(height, image.implicitHeight * viewer.zoom + 48)
            boundsBehavior: Flickable.StopAtBounds
            Rectangle {
                anchors.fill: parent
                color: "#E8EEED"
            }
            Image {
                id: image
                x: Math.max(24, (viewport.contentWidth - width) / 2)
                y: Math.max(24, (viewport.contentHeight - height) / 2)
                source: viewer.source
                width: implicitWidth * viewer.zoom
                height: implicitHeight * viewer.zoom
                fillMode: Image.PreserveAspectFit
                asynchronous: true
                cache: false
            }
            MouseArea {
                anchors.fill: parent
                acceptedButtons: Qt.NoButton
                onWheel: function(event) {
                    const step = event.angleDelta.y > 0 ? 0.12 : -0.12
                    viewer.zoom = Math.max(0.25, Math.min(4.0, viewer.zoom + step))
                    event.accepted = true
                }
            }
            ScrollBar.vertical: ScrollBar {}
            ScrollBar.horizontal: ScrollBar {}
        }
    }
}
