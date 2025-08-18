THEME = """
/* Dark Theme - Cyberpunk inspired */
Screen {
    background: $surface;
}
/* Main containers */
Container {
    background: $panel;
    border: round $primary;
}
/* Headers and titles */
.title {
    color: $text;
    text-style: bold;
    padding: 1;
}
.subtitle {
    color: $text-muted;
    padding: 0 1;
}
/* Dashboard widgets */
.dashboard-widget {
    border: round $primary-lighten-2;
    background: $panel;
    margin: 1;
    padding: 1;
}
.system-monitor {
    border: double $success;
}
.quick-actions {
    border: round $accent;
}
/* Model cards */
.model-card {
    border: round $primary;
    background: $panel;
    margin: 1;
    padding: 1;
    height: 10;
}
.model-card:hover {
    border: double $accent;
    background: $panel-lighten-1;
}
.model-card.selected {
    border: thick $success;
    background: $panel-lighten-2;
}
/* Buttons */
Button {
    margin: 0 1;
    min-width: 16;
}
Button.primary {
    background: $primary;
    color: $text;
}
Button.primary:hover {
    background: $primary-lighten-1;
}
Button.success {
    background: $success;
    color: $text;
}
Button.warning {
    background: $warning;
    color: $text;
}
/* Progress bars */
ProgressBar {
    color: $success;
    background: $panel;
}
.progress-label {
    color: $text-muted;
    text-align: center;
}
/* Input fields */
Input {
    background: $panel-darken-1;
    border: tall $primary;
}
Input:focus {
    border: double $accent;
}
/* Data tables */
DataTable {
    background: $panel;
    color: $text;
}
DataTable > .datatable--header {
    background: $primary;
    color: $text;
    text-style: bold;
}
DataTable > .datatable--cursor {
    background: $accent;
    color: $panel;
}
/* Tabs */
TabbedContent {
    background: $panel;
}
Tab {
    padding: 1 2;
}
Tab.active {
    background: $primary;
    color: $text;
}
/* Lists */
ListView {
    background: $panel;
    border: round $primary;
}
ListItem {
    padding: 0 1;
}
ListItem:hover {
    background: $panel-lighten-1;
}
ListItem.selected {
    background: $primary;
    color: $text;
}
/* Log widgets */
RichLog {
    background: $panel-darken-1;
    border: round $primary-darken-1;
    color: $text;
}
/* Footer */
Footer {
    background: $primary-darken-2;
}
/* Scrollbars */
ScrollBar {
    background: $panel;
    color: $primary;
}
/* Tooltips */
.tooltip {
    background: $panel-lighten-2;
    border: round $accent;
    color: $text;
    padding: 1;
}
/* Loading spinner */
LoadingIndicator {
    color: $accent;
}
/* Success/Error messages */
.success-message {
    color: $success;
    text-style: bold;
}
.error-message {
    color: $error;
    text-style: bold;
}
.warning-message {
    color: $warning;
    text-style: bold;
}
/* Sparkline charts */
.sparkline {
    color: $accent;
    height: 3;
}
/* ASCII art logo */
.logo {
    color: $accent;
    text-align: center;
    text-style: bold;
}