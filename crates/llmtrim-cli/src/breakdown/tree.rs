//! A reusable collapsible tree-table widget for the breakdown TUI.
//!
//! Generic over a per-node payload `D` (what a leaf "is", e.g. a session id or a source
//! block), so the same widget renders the Sessions tree and both Detail panes. The whole
//! tree is materialized up front (the data sets are small and bounded by SQL grouping), so
//! `expanded` just toggles child visibility — no lazy loaders.

use ratatui::Frame;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Cell, Paragraph, Row, Table};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Truncate `s` to at most `max` display columns, appending `…` when it doesn't fit.
pub(crate) fn truncate_w(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.width() <= max {
        return s.to_string();
    }
    let budget = max.saturating_sub(1); // room for the ellipsis
    let mut out = String::new();
    let mut w = 0;
    for ch in s.chars() {
        let cw = ch.width().unwrap_or(0);
        if w + cw > budget {
            break;
        }
        out.push(ch);
        w += cw;
    }
    out.push('…');
    out
}

/// A column definition: header, fixed width, alignment, and an optional value-cell color.
pub struct Column {
    pub header: &'static str,
    pub width: u16,
    pub right: bool,
    /// Foreground color for this column's value cells (not the header). `None` = default.
    pub color: Option<Color>,
}

impl Column {
    pub fn left(header: &'static str, width: u16) -> Self {
        Self {
            header,
            width,
            right: false,
            color: None,
        }
    }
    pub fn right(header: &'static str, width: u16) -> Self {
        Self {
            header,
            width,
            right: true,
            color: None,
        }
    }
    /// Set the value-cell color for this column.
    pub fn colored(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }
}

/// One row in the tree. `cols` are the pre-formatted value columns (after the label).
pub struct TreeNode<D> {
    pub label: String,
    pub cols: Vec<String>,
    pub data: D,
    pub children: Vec<TreeNode<D>>,
    pub expanded: bool,
    pub bold: bool,
}

impl<D> TreeNode<D> {
    pub fn leaf(label: impl Into<String>, cols: Vec<String>, data: D) -> Self {
        Self {
            label: label.into(),
            cols,
            data,
            children: Vec::new(),
            expanded: false,
            bold: false,
        }
    }

    pub fn branch(
        label: impl Into<String>,
        cols: Vec<String>,
        data: D,
        children: Vec<TreeNode<D>>,
    ) -> Self {
        Self {
            label: label.into(),
            cols,
            data,
            children,
            expanded: false,
            bold: false,
        }
    }

    pub fn bold(mut self) -> Self {
        self.bold = true;
        self
    }

    pub fn expanded(mut self) -> Self {
        self.expanded = true;
        self
    }

    fn expandable(&self) -> bool {
        !self.children.is_empty()
    }
}

/// A collapsible tree rendered as a bordered table with a cursor and scroll offset.
pub struct TreeTable<D> {
    pub title: String,
    pub columns: Vec<Column>,
    pub roots: Vec<TreeNode<D>>,
    pub footer: Option<Vec<String>>,
    /// Structural border color (normally `palette::frame()`). Panes are distinguished by their
    /// title, not a hue; focus is shown by bold-vs-dim on this one neutral tone.
    pub border: Color,
    /// Shown centered + dim when the tree is empty (first-run / no data).
    pub empty_hint: String,
    cursor: usize,
    offset: usize,
}

impl<D> TreeTable<D> {
    pub fn new(title: impl Into<String>, columns: Vec<Column>, border: Color) -> Self {
        Self {
            title: title.into(),
            columns,
            roots: Vec::new(),
            footer: None,
            border,
            empty_hint: String::new(),
            cursor: 0,
            offset: 0,
        }
    }

    /// Set the placeholder shown when the tree has no rows.
    pub fn empty_hint(mut self, hint: impl Into<String>) -> Self {
        self.empty_hint = hint.into();
        self
    }

    /// Replace the tree, carrying over which branches the user had expanded (matched by their
    /// label path) and clamping the cursor. This keeps a live refresh from collapsing the
    /// branches the user opened — labels are kept stable by the builders for this reason.
    pub fn set_roots(&mut self, roots: Vec<TreeNode<D>>) {
        let mut expanded = std::collections::HashSet::new();
        collect_expanded(&self.roots, &mut Vec::new(), &mut expanded);
        // Anchor the cursor to the selected node's label path, not its flat index, so a
        // refresh that inserts/reorders rows keeps the selection on the same item.
        let anchor = self.cursor_label_path();
        self.roots = roots;
        if !expanded.is_empty() {
            apply_expanded(&mut self.roots, &mut Vec::new(), &expanded);
        }
        let vis = self.flatten();
        if let Some(anchor) = anchor
            && let Some(i) = vis.iter().position(|(_, p)| self.label_path(p) == anchor)
        {
            self.cursor = i;
        } else if self.cursor >= vis.len() {
            self.cursor = vis.len().saturating_sub(1);
        }
    }

    /// The chain of labels from root to a node at `path` (its stable identity for anchoring).
    fn label_path(&self, path: &[usize]) -> Vec<String> {
        let mut out = Vec::with_capacity(path.len());
        let mut nodes = &self.roots;
        for &i in path {
            let Some(n) = nodes.get(i) else { break };
            out.push(n.label.clone());
            nodes = &n.children;
        }
        out
    }

    /// Label path of the node currently under the cursor, if any.
    fn cursor_label_path(&self) -> Option<Vec<String>> {
        let vis = self.flatten();
        vis.get(self.cursor).map(|(_, p)| self.label_path(p))
    }

    /// Depth + index path of every currently-visible node (pre-order, honoring `expanded`).
    /// The path lets us mutate the node (toggle expand) without borrow gymnastics.
    fn flatten(&self) -> Vec<(usize, Vec<usize>)> {
        fn walk<D>(
            nodes: &[TreeNode<D>],
            depth: usize,
            prefix: &mut Vec<usize>,
            out: &mut Vec<(usize, Vec<usize>)>,
        ) {
            for (i, n) in nodes.iter().enumerate() {
                prefix.push(i);
                out.push((depth, prefix.clone()));
                if n.expanded && !n.children.is_empty() {
                    walk(&n.children, depth + 1, prefix, out);
                }
                prefix.pop();
            }
        }
        let mut out = Vec::new();
        let mut prefix = Vec::new();
        walk(&self.roots, 0, &mut prefix, &mut out);
        out
    }

    fn node_at_path(&self, path: &[usize]) -> Option<&TreeNode<D>> {
        let mut nodes = &self.roots;
        let mut node = None;
        for &i in path {
            let n = nodes.get(i)?;
            node = Some(n);
            nodes = &n.children;
        }
        node
    }

    fn node_at_path_mut(&mut self, path: &[usize]) -> Option<&mut TreeNode<D>> {
        let (&first, rest) = path.split_first()?;
        let mut node = self.roots.get_mut(first)?;
        for &i in rest {
            node = node.children.get_mut(i)?;
        }
        Some(node)
    }

    /// The payload of the node under the cursor, if any.
    pub fn selected(&self) -> Option<&TreeNode<D>> {
        let vis = self.flatten();
        let (_, path) = vis.get(self.cursor)?;
        self.node_at_path(path)
    }

    pub fn move_up(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn move_down(&mut self) {
        let n = self.flatten().len();
        if self.cursor + 1 < n {
            self.cursor += 1;
        }
    }

    pub fn move_page(&mut self, delta: isize) {
        let n = self.flatten().len() as isize;
        let next = (self.cursor as isize + delta).clamp(0, (n - 1).max(0));
        self.cursor = next as usize;
    }

    pub fn jump_top(&mut self) {
        self.cursor = 0;
    }

    pub fn jump_bottom(&mut self) {
        self.cursor = self.flatten().len().saturating_sub(1);
    }

    /// Expand the node under the cursor (or, if a leaf, returns false so the caller can act).
    pub fn expand(&mut self) -> bool {
        let vis = self.flatten();
        let Some((_, path)) = vis.get(self.cursor).cloned() else {
            return false;
        };
        if let Some(n) = self.node_at_path_mut(&path)
            && n.expandable()
        {
            n.expanded = true;
            return true;
        }
        false
    }

    /// Collapse the node under the cursor; returns false if it wasn't an expanded branch.
    pub fn collapse(&mut self) -> bool {
        let vis = self.flatten();
        let Some((_, path)) = vis.get(self.cursor).cloned() else {
            return false;
        };
        if let Some(n) = self.node_at_path_mut(&path)
            && n.expanded
        {
            n.expanded = false;
            return true;
        }
        false
    }

    /// Toggle expand/collapse; for a leaf, returns true so the caller can "activate" it.
    pub fn toggle(&mut self) -> Activate {
        let vis = self.flatten();
        let Some((_, path)) = vis.get(self.cursor).cloned() else {
            return Activate::None;
        };
        match self.node_at_path_mut(&path) {
            Some(n) if n.expandable() => {
                n.expanded = !n.expanded;
                Activate::Toggled
            }
            Some(_) => Activate::Leaf,
            None => Activate::None,
        }
    }

    /// Render into `area`. `focused` draws the cursor row in reverse video and the border
    /// highlighted, so the active pane is obvious in a multi-pane screen.
    pub fn render(&mut self, frame: &mut Frame, area: Rect, focused: bool) {
        // Degenerate area: a box needs at least top+bottom border + 1 row. Below that, a
        // bordered table renders as broken chrome — show a one-line note instead.
        if area.height < 4 || area.width < 8 {
            let note = Paragraph::new(Line::from(format!("{} (too small)", self.title)))
                .style(Style::default().add_modifier(Modifier::DIM));
            frame.render_widget(note, area);
            return;
        }
        let vis = self.flatten();
        let total = vis.len();
        if self.cursor >= total {
            self.cursor = total.saturating_sub(1);
        }

        // Focus = the navigation accent (border + title); a blurred pane recedes to the
        // neutral frame grey, dimmed. This is how the layout says "you are here".
        let (border_style, title_style) = if focused {
            (
                Style::default().fg(super::palette::accent()),
                Style::default().fg(super::palette::accent()),
            )
        } else {
            (
                Style::default()
                    .fg(super::palette::frame())
                    .add_modifier(Modifier::DIM),
                Style::default().fg(super::palette::muted_gray()),
            )
        };
        // Empty tree: render the placeholder hint inside the bordered box.
        if total == 0 {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(border_style)
                .title(format!(" {} ", self.title))
                .title_style(title_style);
            let hint = if self.empty_hint.is_empty() {
                "nothing yet"
            } else {
                self.empty_hint.as_str()
            };
            let p = Paragraph::new(Line::from(hint).centered())
                .style(Style::default().add_modifier(Modifier::DIM))
                .block(block);
            frame.render_widget(p, area);
            return;
        }
        // Rows available inside the border (top + bottom) and the header — and one fewer
        // when a footer row is present, so the data window never overflows the box.
        let footer_rows = u16::from(self.footer.is_some());
        let body_rows = area.height.saturating_sub(3 + footer_rows) as usize;
        if body_rows > 0 {
            if self.cursor < self.offset {
                self.offset = self.cursor;
            } else if self.cursor >= self.offset + body_rows {
                self.offset = self.cursor + 1 - body_rows;
            }
            let max_off = total.saturating_sub(body_rows);
            self.offset = self.offset.min(max_off);
        }

        let mut rows: Vec<Row> = Vec::new();
        let end = (self.offset + body_rows).min(total);
        for (idx, (depth, path)) in vis.iter().enumerate().take(end).skip(self.offset) {
            let Some(node) = self.node_at_path(path) else {
                continue;
            };
            let glyph = if node.expandable() {
                if node.expanded { "▾ " } else { "▸ " }
            } else {
                "  "
            };
            // Truncate the label (indent + glyph + name) to the label column width with an
            // ellipsis, measured by display width so CJK / wide glyphs don't overflow.
            let prefix = format!("{}{}", "  ".repeat(*depth), glyph);
            let label_w = self.columns.first().map_or(40, |c| c.width) as usize;
            let avail = label_w.saturating_sub(prefix.width());
            let name = truncate_w(&node.label, avail);
            let label = format!("{prefix}{name}");
            let mut cells: Vec<Cell> = Vec::with_capacity(node.cols.len() + 1);
            let label_span = if node.bold {
                Span::from(label).bold()
            } else {
                Span::from(label)
            };
            cells.push(Cell::from(Line::from(label_span)));
            for (j, c) in node.cols.iter().enumerate() {
                let mut cell = Cell::from(Line::from(c.clone()).right_aligned());
                // Value columns line up after the label column (index 0).
                if let Some(color) = self.columns.get(j + 1).and_then(|col| col.color) {
                    cell = cell.style(Style::default().fg(color));
                }
                cells.push(cell);
            }
            let mut row = Row::new(cells);
            if focused && idx == self.cursor {
                // A filled swatch (surface1 bg + text fg + bold), not REVERSED: it reads as a
                // raised bar and stays legible over the columns' own fg colors, which inversion
                // would turn into low-contrast mud.
                row = row.style(
                    Style::default()
                        .bg(super::palette::select_bg())
                        .fg(super::palette::selection_fg())
                        .add_modifier(Modifier::BOLD),
                );
            }
            rows.push(row);
        }

        if let Some(footer) = &self.footer {
            let mut cells: Vec<Cell> = Vec::with_capacity(footer.len());
            for (i, c) in footer.iter().enumerate() {
                let line = Line::from(c.clone());
                cells.push(Cell::from(if i == 0 { line } else { line.right_aligned() }).bold());
            }
            rows.push(Row::new(cells).bold());
        }

        let constraints: Vec<Constraint> = self
            .columns
            .iter()
            .map(|c| Constraint::Length(c.width))
            .collect();
        let header = Row::new(
            self.columns
                .iter()
                .map(|c| {
                    let line = Line::from(c.header);
                    Cell::from(if c.right { line.right_aligned() } else { line })
                })
                .collect::<Vec<_>>(),
        )
        .style(
            Style::default()
                .fg(super::palette::muted_gray())
                .add_modifier(Modifier::BOLD),
        );

        // Border is a neutral structural tone (bold when focused, dim otherwise); the title
        // is a plain label. Panes are told apart by their title, not a hue. Scroll arrows in
        // the subtitle signal off-screen content above/below.
        let lo = self.offset + 1;
        let hi = end;
        let up = if self.offset > 0 { "▲" } else { " " };
        let down = if hi < total { "▼" } else { " " };
        let subtitle = format!(" {up} {lo}-{hi}/{total} {down} ");
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(border_style)
            .title(format!(" {} ", self.title))
            .title_style(title_style)
            .title_bottom(Line::from(subtitle).right_aligned());

        let table = Table::new(rows, constraints).header(header).block(block);
        frame.render_widget(table, area);
    }
}

/// Outcome of activating (Enter) the cursor row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activate {
    /// A branch was expanded or collapsed.
    Toggled,
    /// A leaf was activated — the screen should act on its payload.
    Leaf,
    /// Nothing under the cursor.
    None,
}

/// Record the label path of every expanded branch (pre-order, only walking expanded
/// subtrees — a collapsed branch's descendants can't be expanded).
fn collect_expanded<D>(
    nodes: &[TreeNode<D>],
    path: &mut Vec<String>,
    out: &mut std::collections::HashSet<Vec<String>>,
) {
    for n in nodes {
        if n.expanded && !n.children.is_empty() {
            path.push(n.label.clone());
            out.insert(path.clone());
            collect_expanded(&n.children, path, out);
            path.pop();
        }
    }
}

/// Re-expand any branch whose label path was expanded before. Only ever sets `expanded`,
/// so builder-default expansions (e.g. top-level groups) are preserved.
fn apply_expanded<D>(
    nodes: &mut [TreeNode<D>],
    path: &mut Vec<String>,
    set: &std::collections::HashSet<Vec<String>>,
) {
    for n in nodes.iter_mut() {
        path.push(n.label.clone());
        if set.contains(path) {
            n.expanded = true;
        }
        // Only descend into a branch that is (now) expanded — mirroring `collect_expanded`,
        // so a collapsed branch's descendants are never pre-expanded behind it.
        if n.expanded {
            apply_expanded(&mut n.children, path, set);
        }
        path.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> TreeTable<u32> {
        let mut t = TreeTable::new(
            "t",
            vec![Column::left("name", 20), Column::right("n", 6)],
            Color::Green,
        );
        t.set_roots(vec![
            TreeNode::branch(
                "agent",
                vec!["".into()],
                0,
                vec![
                    TreeNode::leaf("s1", vec!["1".into()], 1),
                    TreeNode::leaf("s2", vec!["2".into()], 2),
                ],
            )
            .expanded(),
            TreeNode::leaf("loose", vec!["3".into()], 3),
        ]);
        t
    }

    #[test]
    fn flatten_honors_expanded() {
        let t = sample();
        // agent (expanded) + s1 + s2 + loose = 4 visible.
        assert_eq!(t.flatten().len(), 4);
    }

    #[test]
    fn collapse_hides_children() {
        let mut t = sample();
        assert!(t.collapse()); // cursor on root agent, collapse it
        // agent (collapsed) + loose = 2 visible.
        assert_eq!(t.flatten().len(), 2);
    }

    #[test]
    fn cursor_navigation_and_selection() {
        let mut t = sample();
        t.move_down(); // s1
        t.move_down(); // s2
        assert_eq!(t.selected().map(|n| n.data), Some(2));
        t.jump_bottom();
        assert_eq!(t.selected().map(|n| n.data), Some(3)); // loose
        t.jump_top();
        assert_eq!(t.selected().map(|n| n.data), Some(0)); // agent
    }

    #[test]
    fn set_roots_preserves_user_expansion_by_label() {
        let mut t = sample();
        // Collapse the (builder-expanded) agent branch, then re-expand a fresh equivalent
        // tree manually-expanded state should carry: expand agent, refresh, still expanded.
        t.collapse(); // agent collapsed
        assert_eq!(t.flatten().len(), 2);
        t.expand(); // user re-expands agent
        // Refresh with a freshly-built tree whose agent node is NOT pre-expanded.
        let fresh = vec![
            TreeNode::branch(
                "agent",
                vec!["".into()],
                0,
                vec![TreeNode::leaf("s1", vec!["1".into()], 1)],
            ),
            TreeNode::leaf("loose", vec!["3".into()], 3),
        ];
        t.set_roots(fresh);
        // agent stays expanded across the refresh → agent + s1 + loose = 3 visible.
        assert_eq!(t.flatten().len(), 3);
    }

    #[test]
    fn toggle_reports_leaf_vs_branch() {
        let mut t = sample();
        assert_eq!(t.toggle(), Activate::Toggled); // root branch
        t.expand(); // re-expand
        t.jump_bottom(); // loose leaf
        assert_eq!(t.toggle(), Activate::Leaf);
    }
}
