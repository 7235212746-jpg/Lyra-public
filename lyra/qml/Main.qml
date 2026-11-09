import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

ApplicationWindow {
    id: root
    width: 1360
    height: 820
    minimumWidth: 1080
    minimumHeight: 680
    visible: true
    color: "transparent"
    flags: Qt.Window | Qt.FramelessWindowHint
    title: "Lyra"

    property color glass: "#48F7FBFA"
    property color glassStrong: "#78FFFFFF"
    property color line: "#66FFFFFF"
    property color ink: "#132524"
    property color muted: "#536967"
    property color teal: "#087D77"
    property color tealSoft: "#267DCDC6"
    property color coral: "#D4554B"
    property string folderTarget: "band"
    property string colourTarget: "band"
    property int colourRow: -1

    Component.onCompleted: lyra.startPolling()

    function pageIndex(page) {
        const pages = ["band", "dos", "combined", "fermi", "magnetic", "settings"]
        return Math.max(0, pages.indexOf(page))
    }

    function canPreviewPage(page) {
        if (page === "band") return lyra.bandAvailable
        if (page === "dos") return lyra.dosAvailable
        if (page === "combined") return lyra.combinedAvailable
        if (page === "fermi") return lyra.fermiAvailable
        return false
    }

    function chooseFolder(target, currentPath) {
        folderTarget = target
        if (currentPath && currentPath.length > 0)
            folderDialog.currentFolder = "file:///" + currentPath.replace(/\\/g, "/")
        folderDialog.open()
    }

    function chooseProjectionColour(target, row, currentColour) {
        colourTarget = target
        colourRow = row
        colourDialog.selectedColor = currentColour
        colourDialog.open()
    }

    FolderDialog {
        id: folderDialog
        title: "选择 VASP 计算文件夹"
        onAccepted: lyra.setFolder(root.folderTarget, selectedFolder.toString())
    }

    ColorDialog {
        id: colourDialog
        title: "选择投影颜色"
        onAccepted: lyra.setProjection(root.colourTarget, root.colourRow, "colour", selectedColor.toString())
    }

    Dialog {
        id: errorDialog
        title: "Lyra 无法完成任务"
        modal: true
        anchors.centerIn: parent
        standardButtons: Dialog.Ok
        width: Math.min(480, root.width - 80)
        onAccepted: lyra.clearError()
        onClosed: lyra.clearError()
        contentItem: Text {
            text: lyra.errorText
            color: root.ink
            wrapMode: Text.Wrap
            padding: 18
            font.family: "Microsoft YaHei UI"
        }
    }

    Connections {
        target: lyra
        function onChanged() {
            if (lyra.errorText.length > 0 && !errorDialog.visible)
                errorDialog.open()
        }
    }

    PreviewWindow {
        id: previewWindow
        source: lyra.previewUrl
    }

    Dialog {
        id: qualityDialog
        title: "计算质量报告"
        modal: true
        anchors.centerIn: parent
        standardButtons: Dialog.Close
        width: Math.min(720, root.width - 80)
        height: Math.min(620, root.height - 80)
        ScrollView {
            anchors.fill: parent
            TextArea {
                text: lyra.qualityDetails
                readOnly: true
                wrapMode: TextEdit.Wrap
                color: root.ink
                font.family: "Microsoft YaHei UI"
                font.pixelSize: 12
                background: Rectangle { color: "#F8FBFA" }
            }
        }
    }

    component GlassButton: Button {
        id: control
        implicitHeight: 36
        leftPadding: 14
        rightPadding: 14
        font.family: "Microsoft YaHei UI"
        font.pixelSize: 12
        font.bold: true
        contentItem: Text {
            text: control.text
            color: control.enabled ? (control.highlighted ? "white" : root.ink) : "#83928F"
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font: control.font
        }
        background: Rectangle {
            radius: 7
            color: !control.enabled ? "#22FFFFFF"
                : control.highlighted ? root.teal
                : control.hovered ? "#74FFFFFF" : "#48FFFFFF"
            border.color: control.highlighted ? root.teal : root.line
        }
    }

    component GlassField: TextField {
        id: field
        implicitHeight: 36
        color: root.ink
        selectedTextColor: "white"
        selectionColor: root.teal
        font.family: "Segoe UI Variable Text"
        font.pixelSize: 12
        leftPadding: 10
        rightPadding: 10
        background: Rectangle {
            radius: 6
            color: "#56FFFFFF"
            border.color: field.activeFocus ? root.teal : root.line
            border.width: field.activeFocus ? 1.5 : 1
        }
    }

    component GlassCombo: ComboBox {
        id: combo
        implicitHeight: 36
        leftPadding: 10
        rightPadding: 30
        font.family: "Microsoft YaHei UI"
        font.pixelSize: 12
        contentItem: Text {
            text: combo.displayText
            color: combo.enabled ? root.ink : "#83928F"
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
            leftPadding: 2
        }
        background: Rectangle {
            radius: 6
            color: combo.enabled ? "#56FFFFFF" : "#22FFFFFF"
            border.color: combo.activeFocus ? root.teal : root.line
        }
    }

    component SectionTitle: Text {
        color: root.ink
        font.family: "Microsoft YaHei UI"
        font.pixelSize: 13
        font.bold: true
    }

    component MutedText: Text {
        color: root.muted
        font.family: "Microsoft YaHei UI"
        font.pixelSize: 11
        wrapMode: Text.Wrap
    }

    component FolderPicker: Rectangle {
        id: picker
        property string target
        property string path
        Layout.fillWidth: true
        implicitHeight: 52
        radius: 7
        color: "#42FFFFFF"
        border.color: root.line
        RowLayout {
            anchors.fill: parent
            anchors.margins: 8
            Text {
                Layout.fillWidth: true
                text: picker.path.length > 0 ? picker.path : "尚未选择文件夹"
                color: picker.path.length > 0 ? root.ink : root.muted
                elide: Text.ElideMiddle
                font.family: "Segoe UI Variable Text"
                font.pixelSize: 12
            }
            GlassButton {
                text: "选择文件夹"
                highlighted: true
                onClicked: root.chooseFolder(picker.target, picker.path)
            }
        }
    }

    component CapabilityBox: Rectangle {
        id: box
        property string report
        Layout.fillWidth: true
        implicitHeight: Math.max(78, reportText.implicitHeight + 22)
        radius: 7
        color: "#36FFFFFF"
        border.color: root.line
        Text {
            id: reportText
            anchors.fill: parent
            anchors.margins: 11
            text: box.report
            color: root.ink
            font.family: "Microsoft YaHei UI"
            font.pixelSize: 11
            lineHeight: 1.32
            wrapMode: Text.Wrap
        }
    }

    component ProjectionEditor: Rectangle {
        id: editor
        property string target
        property var channelModel
        property var elements: []
        property bool editorEnabled: true
        Layout.fillWidth: true
        implicitHeight: editorLayout.implicitHeight + 22
        radius: 7
        color: "#30FFFFFF"
        border.color: root.line
        ColumnLayout {
            id: editorLayout
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: 11
            spacing: 7
            RowLayout {
                Layout.fillWidth: true
                SectionTitle { text: "投影通道" }
                Item { Layout.fillWidth: true }
                GlassButton {
                    text: "+ 添加"
                    enabled: editor.editorEnabled
                    onClicked: lyra.addProjection(editor.target)
                }
            }
            MutedText {
                Layout.fillWidth: true
                text: editor.channelModel.count === 0
                    ? "添加元素或轨道通道后即可绘制投影。"
                    : (editor.channelModel.rgbCompatible ? "" : "已超过 3 条通道，RGB 预览会自动切换为 Fat band。")
                visible: text.length > 0
            }
            Repeater {
                model: editor.channelModel
                delegate: RowLayout {
                    id: projectionRow
                    required property int index
                    required property bool selected
                    required property string element
                    required property string orbital
                    required property string colour
                    Layout.fillWidth: true
                    spacing: 6
                    CheckBox {
                        checked: projectionRow.selected
                        enabled: editor.editorEnabled
                        onToggled: lyra.setProjection(editor.target, projectionRow.index, "enabled", checked)
                    }
                    GlassCombo {
                        Layout.fillWidth: true
                        enabled: editor.editorEnabled
                        model: ["全部元素"].concat(editor.elements)
                        currentIndex: Math.max(0, model.indexOf(projectionRow.element))
                        onActivated: lyra.setProjection(editor.target, projectionRow.index, "element", currentText)
                    }
                    GlassCombo {
                        Layout.preferredWidth: 102
                        enabled: editor.editorEnabled
                        model: ["全部", "s", "p", "d", "f", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2"]
                        currentIndex: Math.max(0, model.indexOf(projectionRow.orbital))
                        onActivated: lyra.setProjection(editor.target, projectionRow.index, "orbital", currentText)
                    }
                    Button {
                        implicitWidth: 34; implicitHeight: 34
                        enabled: editor.editorEnabled
                        ToolTip.visible: hovered
                        ToolTip.text: "选择颜色"
                        onClicked: root.chooseProjectionColour(editor.target, projectionRow.index, projectionRow.colour)
                        background: Rectangle {
                            radius: 6
                            color: projectionRow.colour
                            border.color: "#70FFFFFF"
                            border.width: 2
                        }
                    }
                    ToolButton {
                        text: "×"
                        enabled: editor.editorEnabled
                        ToolTip.visible: hovered
                        ToolTip.text: "删除通道"
                        onClicked: lyra.removeProjection(editor.target, projectionRow.index)
                    }
                }
            }
        }
    }

    component RangeRow: RowLayout {
        id: range
        property string label
        property string minimumText
        property string maximumText
        property string minimumKey
        property string maximumKey
        spacing: 8
        Text { text: range.label; color: root.ink; font.family: "Microsoft YaHei UI"; font.pixelSize: 12 }
        Item { Layout.fillWidth: true }
        GlassField {
            Layout.preferredWidth: 78
            text: range.minimumText
            horizontalAlignment: Text.AlignHCenter
            onEditingFinished: lyra.setValue(range.minimumKey, text)
        }
        Text { text: "至"; color: root.muted }
        GlassField {
            Layout.preferredWidth: 78
            text: range.maximumText
            horizontalAlignment: Text.AlignHCenter
            onEditingFinished: lyra.setValue(range.maximumKey, text)
        }
    }

    component PageHeading: RowLayout {
        id: heading
        property string title
        property string subtitle
        Layout.fillWidth: true
        ColumnLayout {
            spacing: 2
            Text { text: heading.title; color: root.ink; font.family: "Microsoft YaHei UI"; font.pixelSize: 20; font.bold: true }
            MutedText { text: heading.subtitle }
        }
        Item { Layout.fillWidth: true }
        Rectangle {
            width: 86; height: 30; radius: 7
            color: "#38FFFFFF"; border.color: root.line
            Text {
                anchors.centerIn: parent
                text: lyra.readMode === "lite" ? "Lite Mode" : "Full Mode"
                color: root.teal; font.bold: true; font.pixelSize: 11
            }
        }
    }

    component BandPage: ScrollView {
        id: bandScroll
        clip: true
        ColumnLayout {
            width: bandScroll.availableWidth
            spacing: 13
            PageHeading { title: "能带"; subtitle: "从显式选择的 Band 文件夹读取" }
            SectionTitle { text: lyra.readMode === "lite" ? "EIGENVAL + KPOINTS + POSCAR + OUTCAR" : "vasprun.xml(.gz) + KPOINTS" }
            FolderPicker { target: "band"; path: lyra.bandFolder }
            CapabilityBox { report: lyra.bandStatus }
            RangeRow {
                label: "能量范围 (eV)"
                minimumText: lyra.bandEnergyMin; maximumText: lyra.bandEnergyMax
                minimumKey: "bandEnergyMin"; maximumKey: "bandEnergyMax"
            }
            Rectangle {
                Layout.fillWidth: true; implicitHeight: 112; radius: 7
                color: "#30FFFFFF"; border.color: root.line
                ColumnLayout {
                    anchors.fill: parent; anchors.margins: 11; spacing: 7
                    SectionTitle { text: "能带数据" }
                    RowLayout {
                        RadioButton {
                            text: "完整能带"
                            checked: !lyra.bandProjected
                            onClicked: lyra.setBoolValue("bandProjected", false)
                        }
                        RadioButton {
                            text: "元素 / 轨道投影"
                            checked: lyra.bandProjected
                            enabled: lyra.bandProjectionAvailable
                            onClicked: lyra.setBoolValue("bandProjected", true)
                        }
                    }
                    MutedText { text: lyra.bandProjectionReason; Layout.fillWidth: true }
                }
            }
            RowLayout {
                Layout.fillWidth: true
                visible: lyra.bandProjected
                Text { text: "投影画法"; color: root.ink }
                Item { Layout.fillWidth: true }
                GlassCombo {
                    model: ["RGB", "Fat band"]
                    currentIndex: lyra.bandProjectionStyle === "RGB" ? 0 : 1
                    onActivated: lyra.setValue("bandProjectionStyle", currentText)
                }
            }
            ProjectionEditor {
                visible: lyra.bandProjected
                target: "band"
                channelModel: lyra.bandProjectionModel
                elements: lyra.bandElements
                editorEnabled: lyra.bandProjectionAvailable
            }
            Item { Layout.fillHeight: true; Layout.minimumHeight: 6 }
        }
    }

    component DosPage: ScrollView {
        id: dosScroll
        clip: true
        ColumnLayout {
            width: dosScroll.availableWidth
            spacing: 13
            PageHeading { title: "态密度"; subtitle: "DOS / PDOS 从独立的 DOS 或 SCF 文件夹读取" }
            SectionTitle { text: lyra.readMode === "lite" ? "DOSCAR + POSCAR + OUTCAR" : "vasprun.xml(.gz)" }
            FolderPicker { target: "dos"; path: lyra.dosFolder }
            CapabilityBox { report: lyra.dosStatus }
            RangeRow {
                label: "能量范围 (eV)"
                minimumText: lyra.dosEnergyMin; maximumText: lyra.dosEnergyMax
                minimumKey: "dosEnergyMin"; maximumKey: "dosEnergyMax"
            }
            RangeRow {
                label: "DOS 范围（留空自动）"
                minimumText: lyra.dosDensityMin; maximumText: lyra.dosDensityMax
                minimumKey: "dosDensityMin"; maximumKey: "dosDensityMax"
            }
            CheckBox {
                text: "绘制元素 / 轨道投影"
                checked: lyra.dosProjected
                enabled: lyra.dosProjectionAvailable
                onToggled: lyra.setBoolValue("dosProjected", checked)
            }
            MutedText {
                text: lyra.dosProjectionAvailable ? "已检测到可用的 PDOS 数据。" : "当前数据没有可用的投影块，投影选项已禁用。"
                Layout.fillWidth: true
            }
            ProjectionEditor {
                visible: lyra.dosProjected
                target: "dos"
                channelModel: lyra.dosProjectionModel
                elements: lyra.dosElements
                editorEnabled: lyra.dosProjectionAvailable
            }
            Item { Layout.fillHeight: true; Layout.minimumHeight: 6 }
        }
    }

    component CombinedPage: ScrollView {
        id: combinedScroll
        clip: true
        ColumnLayout {
            width: combinedScroll.availableWidth
            spacing: 13
            PageHeading { title: "能带 + 态密度"; subtitle: "Band 与 DOS 文件夹分别选择" }
            RowLayout {
                Layout.fillWidth: true
                SectionTitle { text: "Band 文件夹" }
                Item { Layout.fillWidth: true }
                GlassCombo {
                    model: ["Lite", "Full"]
                    currentIndex: lyra.combinedBandReadMode === "lite" ? 0 : 1
                    onActivated: lyra.setValue("combinedBandReadMode", currentIndex === 0 ? "lite" : "full")
                }
            }
            FolderPicker { target: "combinedBand"; path: lyra.combinedBandFolder }
            CapabilityBox { report: lyra.combinedBandStatus }
            RowLayout {
                Layout.fillWidth: true
                SectionTitle { text: "DOS / SCF 文件夹" }
                Item { Layout.fillWidth: true }
                GlassCombo {
                    model: ["Lite", "Full"]
                    currentIndex: lyra.combinedDosReadMode === "lite" ? 0 : 1
                    onActivated: lyra.setValue("combinedDosReadMode", currentIndex === 0 ? "lite" : "full")
                }
            }
            FolderPicker { target: "combinedDos"; path: lyra.combinedDosFolder }
            CapabilityBox { report: lyra.combinedDosStatus }
            Rectangle {
                Layout.fillWidth: true; implicitHeight: 38; radius: 7
                color: root.tealSoft; border.color: "#5CA9C4C1"
                Text { anchors.centerIn: parent; text: lyra.combinedReference; color: root.teal; font.bold: true; font.pixelSize: 11 }
            }
            RangeRow {
                label: "共同能量范围 (eV)"
                minimumText: lyra.combinedEnergyMin; maximumText: lyra.combinedEnergyMax
                minimumKey: "combinedEnergyMin"; maximumKey: "combinedEnergyMax"
            }
            RangeRow {
                label: "DOS 横轴（留空自动）"
                minimumText: lyra.combinedDosDensityMin; maximumText: lyra.combinedDosDensityMax
                minimumKey: "combinedDosDensityMin"; maximumKey: "combinedDosDensityMax"
            }
            RowLayout {
                Layout.fillWidth: true
                CheckBox {
                    text: "投影能带"
                    checked: lyra.combinedBandProjected
                    enabled: lyra.combinedBandProjectionAvailable
                    onToggled: lyra.setBoolValue("combinedBandProjected", checked)
                }
                CheckBox {
                    text: "投影 DOS"
                    checked: lyra.combinedDosProjected
                    enabled: lyra.combinedDosProjectionAvailable
                    onToggled: lyra.setBoolValue("combinedDosProjected", checked)
                }
                Item { Layout.fillWidth: true }
                GlassCombo {
                    visible: lyra.combinedBandProjected
                    model: ["RGB", "Fat band"]
                    currentIndex: lyra.combinedProjectionStyle === "RGB" ? 0 : 1
                    onActivated: lyra.setValue("combinedProjectionStyle", currentText)
                }
            }
            ProjectionEditor {
                visible: lyra.combinedBandProjected || lyra.combinedDosProjected
                target: "combined"
                channelModel: lyra.combinedProjectionModel
                elements: lyra.combinedElements
                editorEnabled: lyra.combinedBandProjectionAvailable || lyra.combinedDosProjectionAvailable
            }
        }
    }

    component FermiPage: ScrollView {
        id: fermiScroll
        clip: true
        ColumnLayout {
            width: fermiScroll.availableWidth
            spacing: 13
            PageHeading { title: "费米面"; subtitle: "需要完整三维均匀 k 网格" }
            SectionTitle { text: lyra.readMode === "lite" ? "EIGENVAL + KPOINTS + POSCAR + OUTCAR" : "vasprun.xml(.gz)" }
            FolderPicker { target: "fermi"; path: lyra.fermiFolder }
            CapabilityBox { report: lyra.fermiStatus }
            RowLayout {
                Layout.fillWidth: true
                Text { text: "视图"; color: root.ink }
                Item { Layout.fillWidth: true }
                GlassCombo {
                    model: ["三维表面", "二维切片"]
                    currentIndex: lyra.fermiViewMode === "surface" ? 0 : 1
                    onActivated: lyra.setValue("fermiViewMode", currentIndex === 0 ? "surface" : "slice")
                }
            }
            RowLayout {
                Layout.fillWidth: true
                Text { text: "相对 E-fermi 偏移 (eV)"; color: root.ink }
                Item { Layout.fillWidth: true }
                GlassField { Layout.preferredWidth: 92; text: lyra.fermiEnergyOffset; onEditingFinished: lyra.setValue("fermiEnergyOffset", text) }
            }
            MutedText { text: "Line-mode KPOINTS 会被识别为路径计算并禁用费米面绘制。"; Layout.fillWidth: true }
        }
    }

    component MagneticPage: ScrollView {
        id: magneticScroll
        clip: true
        ColumnLayout {
            width: magneticScroll.availableWidth
            spacing: 13
            PageHeading { title: "磁矩结构"; subtitle: "导出可在 VESTA 中显示磁矩方向的结构文件" }
            SectionTitle { text: "POSCAR + OUTCAR" }
            FolderPicker { target: "magnetic"; path: lyra.magneticFolder }
            CapabilityBox { report: lyra.magneticStatus }
            MutedText { text: "输出位置：" + lyra.outputFolder; Layout.fillWidth: true }
            GlassButton {
                text: "导出 VESTA 磁矩结构"
                highlighted: true
                enabled: lyra.canExportMagnetic && !lyra.busy
                onClicked: lyra.exportMagnetic()
            }
            Repeater {
                model: lyra.magneticOutputFiles
                delegate: MutedText { required property string modelData; text: modelData; Layout.fillWidth: true }
            }
            Item { Layout.fillHeight: true }
        }
    }

    component SettingsPage: ScrollView {
        id: settingsScroll
        clip: true
        ColumnLayout {
            width: settingsScroll.availableWidth
            spacing: 13
            PageHeading { title: "设置"; subtitle: "关闭 Lyra 时自动保存" }
            SectionTitle { text: "导出" }
            FolderPicker { target: "output"; path: lyra.outputFolder }
            RowLayout {
                Layout.fillWidth: true
                Text { text: "PNG DPI"; color: root.ink }
                Item { Layout.fillWidth: true }
                GlassField { Layout.preferredWidth: 92; text: lyra.dpi; onEditingFinished: lyra.setValue("dpi", text) }
            }
            RowLayout {
                Layout.fillWidth: true
                Text { text: "图宽 (mm)"; color: root.ink }
                Item { Layout.fillWidth: true }
                GlassField { Layout.preferredWidth: 92; text: lyra.widthMm; onEditingFinished: lyra.setValue("widthMm", text) }
            }
        }
    }

    Rectangle {
        anchors.fill: parent
        color: "#18EDF5F3"
        border.color: root.line
        border.width: 1
        radius: 10
    }

    Rectangle {
        id: titleBar
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        height: 54
        color: "#30FFFFFF"
        topLeftRadius: 10
        topRightRadius: 10
        border.color: root.line
        MouseArea {
            anchors.fill: parent
            onPressed: root.startSystemMove()
            onDoubleClicked: root.visibility === Window.Maximized ? root.showNormal() : root.showMaximized()
        }
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 18
            spacing: 10
            Rectangle {
                width: 28; height: 28; radius: 7; color: root.teal
                Text { anchors.centerIn: parent; text: "L"; color: "white"; font.bold: true; font.pixelSize: 15 }
            }
            Text { text: "Lyra"; color: root.ink; font.family: "Segoe UI Variable Display"; font.pixelSize: 22; font.bold: true }
            Text { text: "电子结构工作台"; color: root.muted; font.family: "Microsoft YaHei UI"; font.pixelSize: 12 }
            Item { Layout.fillWidth: true }
            Rectangle {
                width: 162; height: 32; radius: 8; color: "#48FFFFFF"; border.color: root.line
                Row {
                    anchors.centerIn: parent; spacing: 4
                    Repeater {
                        model: [{label: "Lite", value: "lite"}, {label: "Full", value: "full"}]
                        delegate: Rectangle {
                            required property var modelData
                            width: 74; height: 26; radius: 6
                            color: lyra.readMode === modelData.value ? root.teal : "transparent"
                            Text { anchors.centerIn: parent; text: modelData.label; color: lyra.readMode === modelData.value ? "white" : root.muted; font.bold: true }
                            MouseArea { anchors.fill: parent; onClicked: lyra.setValue("readMode", modelData.value) }
                        }
                    }
                }
            }
            ToolButton { text: "-"; onClicked: root.showMinimized(); background: Rectangle { color: parent.hovered ? "#40FFFFFF" : "transparent" } }
            ToolButton { text: root.visibility === Window.Maximized ? "❐" : "□"; onClicked: root.visibility === Window.Maximized ? root.showNormal() : root.showMaximized(); background: Rectangle { color: parent.hovered ? "#40FFFFFF" : "transparent" } }
            ToolButton { text: "×"; onClicked: root.close(); background: Rectangle { color: parent.hovered ? root.coral : "transparent" } }
        }
    }

    RowLayout {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: titleBar.bottom
        anchors.bottom: parent.bottom
        anchors.margins: 14
        spacing: 12

        Rectangle {
            Layout.preferredWidth: 190
            Layout.fillHeight: true
            radius: 8
            color: "#48FFFFFF"
            border.color: root.line
            Column {
                anchors.fill: parent
                anchors.margins: 14
                spacing: 7
                Text { text: "分析"; color: root.muted; font.pixelSize: 11; font.bold: true }
                Repeater {
                    model: [
                        {key: "band", label: "能带", enabled: true},
                        {key: "dos", label: "态密度", enabled: true},
                        {key: "combined", label: "能带 + 态密度", enabled: true},
                        {key: "fermi", label: "费米面", enabled: true},
                        {key: "magnetic", label: "磁矩结构", enabled: true}
                    ]
                    delegate: Rectangle {
                        required property var modelData
                        width: 162; height: 38; radius: 7
                        color: lyra.currentPage === modelData.key ? "#78FFFFFF" : "transparent"
                        border.color: lyra.currentPage === modelData.key ? "#7CA9C4C1" : "transparent"
                        Text {
                            anchors.left: parent.left; anchors.leftMargin: 12; anchors.verticalCenter: parent.verticalCenter
                            text: modelData.label; color: lyra.currentPage === modelData.key ? root.teal : root.ink
                            font.family: "Microsoft YaHei UI"; font.pixelSize: 13; font.bold: lyra.currentPage === modelData.key
                        }
                        MouseArea { anchors.fill: parent; onClicked: lyra.setValue("currentPage", modelData.key) }
                    }
                }
                Item { width: 1; height: 10 }
                Text { text: "规划中"; color: root.muted; font.pixelSize: 11; font.bold: true }
                Repeater {
                    model: ["成键分析", "价态与电荷", "远程连接"]
                    delegate: Rectangle {
                        required property string modelData
                        width: 162; height: 34; radius: 7; color: "#28FFFFFF"
                        Text { anchors.left: parent.left; anchors.leftMargin: 12; anchors.verticalCenter: parent.verticalCenter; text: modelData; color: "#82918F"; font.family: "Microsoft YaHei UI"; font.pixelSize: 12 }
                    }
                }
                Item { width: 1; height: 10 }
                Rectangle {
                    width: 162; height: 38; radius: 7
                    color: lyra.currentPage === "settings" ? "#78FFFFFF" : "transparent"
                    Text { anchors.left: parent.left; anchors.leftMargin: 12; anchors.verticalCenter: parent.verticalCenter; text: "设置"; color: lyra.currentPage === "settings" ? root.teal : root.ink; font.bold: lyra.currentPage === "settings" }
                    MouseArea { anchors.fill: parent; onClicked: lyra.setValue("currentPage", "settings") }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: 8
            color: "#58FFFFFF"
            border.color: root.line
            RowLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 14
                StackLayout {
                    Layout.preferredWidth: 455
                    Layout.minimumWidth: 390
                    Layout.fillHeight: true
                    currentIndex: root.pageIndex(lyra.currentPage)
                    BandPage {}
                    DosPage {}
                    CombinedPage {}
                    FermiPage {}
                    MagneticPage {}
                    SettingsPage {}
                }
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumWidth: 360
                    radius: 7
                    color: "#F6FFFFFF"
                    border.color: "#7CA9C4C1"
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 10
                        RowLayout {
                            Layout.fillWidth: true
                            SectionTitle { text: "预览与输出" }
                            Item { Layout.fillWidth: true }
                            MutedText { text: lyra.readMode === "lite" ? "Lite Mode" : "Full Mode" }
                        }
                        Rectangle {
                            Layout.fillWidth: true; Layout.fillHeight: true
                            color: "white"; border.color: "#D3DEDC"; radius: 4
                            Image {
                                anchors.fill: parent
                                anchors.margins: 6
                                source: lyra.previewUrl
                                fillMode: Image.PreserveAspectFit
                                asynchronous: true
                                cache: false
                                visible: lyra.previewUrl.length > 0
                            }
                            Text {
                                anchors.centerIn: parent
                                text: "选择文件夹后即可生成论文图预览"
                                color: root.muted
                                font.pixelSize: 13
                                visible: lyra.previewUrl.length === 0
                            }
                            BusyIndicator {
                                anchors.centerIn: parent
                                running: lyra.busy
                                visible: running
                            }
                        }
                        MutedText {
                            Layout.fillWidth: true
                            text: lyra.qualitySummary
                            visible: lyra.previewUrl.length > 0
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            visible: lyra.previewUrl.length > 0
                            GlassButton { text: "打开图片"; onClicked: lyra.openPreviewExternal() }
                            GlassButton { text: "放大预览"; onClicked: previewWindow.open() }
                            GlassButton { text: "打开文件夹"; onClicked: lyra.openOutputFolder() }
                            Item { Layout.fillWidth: true }
                            GlassButton {
                                text: "质量报告"
                                enabled: lyra.qualityAvailable
                                onClicked: qualityDialog.open()
                            }
                        }
                        Repeater {
                            model: lyra.outputFiles
                            delegate: Text {
                                required property string modelData
                                Layout.fillWidth: true
                                text: modelData
                                color: root.muted
                                elide: Text.ElideMiddle
                                font.pixelSize: 10
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            Text { text: lyra.statusText; color: root.teal; font.bold: true; elide: Text.ElideRight; Layout.fillWidth: true }
                            GlassButton {
                                text: "预览"
                                highlighted: true
                                enabled: root.canPreviewPage(lyra.currentPage) && !lyra.busy
                                onClicked: lyra.previewCurrent()
                            }
                            GlassButton {
                                text: "保存图片"
                                enabled: lyra.canSave && !lyra.busy
                                onClicked: lyra.saveCurrent()
                            }
                        }
                    }
                }
            }
        }
    }
}
