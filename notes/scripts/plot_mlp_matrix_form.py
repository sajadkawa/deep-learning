import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 11)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title("MLP Matrix Form", fontsize=13, fontweight='bold', pad=12)

fs   = 11
fss  = 8.5
mono = 'DejaVu Sans Mono'

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Matrix definitions
# ─────────────────────────────────────────────────────────────────────────────
ax.text(0.2, 8.7, "Matrices", fontsize=11, fontweight='bold', color='black')
ax.axhline(y=8.5, xmin=0.02, xmax=0.98, color='lightgray', lw=0.8)

# W⁽¹⁾ at x=0.3
ax.text(0.3, 8.1, "⎡w⁽¹⁾₁₁  w⁽¹⁾₁₂⎤", fontsize=fs, family=mono, va='center')
ax.text(0.3, 7.5, "⎣w⁽¹⁾₂₁  w⁽¹⁾₂₂⎦", fontsize=fs, family=mono, va='center')
ax.text(1.15, 8.45, "W⁽¹⁾", fontsize=10, color='dimgray', ha='center', style='italic')
ax.text(1.15, 7.15, "shape (n×m)", fontsize=fss, color='gray', ha='center', style='italic')

# x at x=3.0
ax.text(3.0, 8.1, "⎡x₁⎤", fontsize=fs, family=mono, va='center')
ax.text(3.0, 7.5, "⎣x₂⎦", fontsize=fs, family=mono, va='center')
ax.text(3.25, 8.45, "x", fontsize=10, color='dimgray', ha='center', style='italic')
ax.text(3.25, 7.15, "shape (n×1)", fontsize=fss, color='gray', ha='center', style='italic')

# b⁽¹⁾ at x=4.5
ax.text(4.5, 8.1, "⎡b₁⁽¹⁾⎤", fontsize=fs, family=mono, va='center')
ax.text(4.5, 7.5, "⎣b₂⁽¹⁾⎦", fontsize=fs, family=mono, va='center')
ax.text(4.85, 8.45, "b⁽¹⁾", fontsize=10, color='dimgray', ha='center', style='italic')
ax.text(4.85, 7.15, "shape (m×1)", fontsize=fss, color='gray', ha='center', style='italic')

# W⁽²⁾ at x=6.3
ax.text(6.3, 8.1, "⎡w⁽²⁾₁₁⎤", fontsize=fs, family=mono, va='center')
ax.text(6.3, 7.5, "⎣w⁽²⁾₂₁⎦", fontsize=fs, family=mono, va='center')
ax.text(6.7, 8.45, "W⁽²⁾", fontsize=10, color='dimgray', ha='center', style='italic')
ax.text(6.7, 7.15, "shape (m×1)", fontsize=fss, color='gray', ha='center', style='italic')

# b⁽²⁾ at x=8.0
ax.text(8.0, 7.8, "b⁽²⁾", fontsize=fs, family=mono, va='center')
ax.text(8.2, 8.45, "b⁽²⁾", fontsize=10, color='dimgray', ha='center', style='italic')
ax.text(8.2, 7.15, "scalar", fontsize=fss, color='gray', ha='center', style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Layer 1
# ─────────────────────────────────────────────────────────────────────────────
ax.text(0.2, 6.7, "Layer 1 (Input → Hidden)", fontsize=11, fontweight='bold', color='goldenrod')
ax.axhline(y=6.5, xmin=0.02, xmax=0.98, color='lightgray', lw=0.8)

L1_top = 6.1
L1_bot = 5.4
L1_mid = (L1_top + L1_bot) / 2
L1_shp = L1_bot - 0.3

# W⁽¹⁾ᵀ  (note: off-diagonal elements are transposed: w₂₁ and w₁₂ swap positions)
ax.text(0.3,  L1_top, "⎡w⁽¹⁾₁₁  w⁽¹⁾₂₁⎤", fontsize=fs, family=mono, va='center')
ax.text(0.3,  L1_bot, "⎣w⁽¹⁾₁₂  w⁽¹⁾₂₂⎦", fontsize=fs, family=mono, va='center')
ax.text(1.15, L1_top + 0.25, "W⁽¹⁾ᵀ", fontsize=9, color='dimgray', ha='center', style='italic')
ax.text(1.15, L1_shp, "(n×m)ᵀ=(m×n)", fontsize=fss, color='gray', ha='center', style='italic')

# ·
ax.text(2.55, L1_mid, "·", fontsize=18, va='center', ha='center')

# x
ax.text(2.75, L1_top, "⎡x₁⎤", fontsize=fs, family=mono, va='center')
ax.text(2.75, L1_bot, "⎣x₂⎦", fontsize=fs, family=mono, va='center')
ax.text(3.05, L1_top + 0.25, "x", fontsize=9, color='dimgray', ha='center', style='italic')
ax.text(3.05, L1_shp, "(n×1)", fontsize=fss, color='gray', ha='center', style='italic')

# +
ax.text(3.55, L1_mid, "+", fontsize=fs, va='center', ha='center')

# b⁽¹⁾
ax.text(3.8,  L1_top, "⎡b₁⁽¹⁾⎤", fontsize=fs, family=mono, va='center')
ax.text(3.8,  L1_bot, "⎣b₂⁽¹⁾⎦", fontsize=fs, family=mono, va='center')
ax.text(4.2,  L1_top + 0.25, "b⁽¹⁾", fontsize=9, color='dimgray', ha='center', style='italic')
ax.text(4.2,  L1_shp, "(m×1)", fontsize=fss, color='gray', ha='center', style='italic')

# = z⁽¹⁾ → f(·) →
ax.text(4.75, L1_mid, "= z⁽¹⁾  →  f( · )  →", fontsize=fs, va='center')

# h
ax.text(7.2,  L1_top, "⎡h₁⎤", fontsize=fs, family=mono, va='center')
ax.text(7.2,  L1_bot, "⎣h₂⎦", fontsize=fs, family=mono, va='center')
ax.text(7.5,  L1_top + 0.25, "h", fontsize=9, color='dimgray', ha='center', style='italic')
ax.text(7.5,  L1_shp, "(m×1)", fontsize=fss, color='gray', ha='center', style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Layer 2
# ─────────────────────────────────────────────────────────────────────────────
ax.text(0.2, 4.6, "Layer 2 (Hidden → Output)", fontsize=11, fontweight='bold', color='seagreen')
ax.axhline(y=4.4, xmin=0.02, xmax=0.98, color='lightgray', lw=0.8)

L2_top = 3.9
L2_bot = 3.2
L2_mid = (L2_top + L2_bot) / 2
L2_shp = L2_bot - 0.3

# W⁽²⁾ᵀ — row vector (1×m)
ax.text(0.3,  L2_mid, "[w⁽²⁾₁₁  w⁽²⁾₂₁]", fontsize=fs, family=mono, va='center')
ax.text(1.2,  L2_top + 0.25, "W⁽²⁾ᵀ", fontsize=9, color='dimgray', ha='center', style='italic')
ax.text(1.2,  L2_shp, "(m×1)ᵀ=(1×m)", fontsize=fss, color='gray', ha='center', style='italic')

# ·
ax.text(2.25, L2_mid, "·", fontsize=18, va='center', ha='center')

# h — column vector (m×1)
ax.text(2.5,  L2_top, "⎡h₁⎤", fontsize=fs, family=mono, va='center')
ax.text(2.5,  L2_bot, "⎣h₂⎦", fontsize=fs, family=mono, va='center')
ax.text(2.8,  L2_top + 0.25, "h", fontsize=9, color='dimgray', ha='center', style='italic')
ax.text(2.8,  L2_shp, "(m×1)", fontsize=fss, color='gray', ha='center', style='italic')

# +
ax.text(3.3,  L2_mid, "+", fontsize=fs, va='center', ha='center')

# b⁽²⁾ scalar
ax.text(3.55, L2_mid, "b⁽²⁾", fontsize=fs, family=mono, va='center')
ax.text(3.75, L2_top + 0.25, "b⁽²⁾", fontsize=9, color='dimgray', ha='center', style='italic')
ax.text(3.75, L2_shp, "scalar", fontsize=fss, color='gray', ha='center', style='italic')

# = z⁽²⁾ → f(·) → ŷ
ax.text(4.35, L2_mid, "= z⁽²⁾  →  f( · )  →  ŷ", fontsize=fs, va='center')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'mlp_matrix_form.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to {os.path.abspath(output_path)}")

