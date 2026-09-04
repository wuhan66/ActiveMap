_base_ = [
    "/home/wh/ActiveMap/external/open_cd/configs/ban/"
    "ban_vit-b16-clip_mit-b0_512x512_40k_levircd.py"
]

model = dict(
    decode_head=dict(
        ban_cfg=dict(
            side_enc_cfg=dict(
                init_cfg=dict(
                    type="Pretrained",
                    checkpoint=(
                        "/home/wh/ActiveMap/models/opencd_ban/"
                        "mit_b0_20220624-7e0fe6dd.pth"
                    ),
                )
            )
        )
    )
)
