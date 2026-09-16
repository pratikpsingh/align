"""Explicit visual-only localization of the pinned Hummingbird USD asset."""

import hashlib


def localize_materials(source, destination):
    """Remove external shader materials; retain every nonmaterial authored value.

    This deliberately changes appearance to untextured geometry. It is neither
    a missing-asset fallback nor a modification of the upstream source file.
    """
    from pxr import Sdf, Usd

    original = Sdf.Layer.FindOrOpen(str(source))
    if original is None:
        raise RuntimeError(f"Cannot open drone model: {source}")
    layer = Sdf.Layer.CreateAnonymous("align-local-visuals.usda")
    layer.TransferContent(original)
    stage = Usd.Stage.Open(layer)
    material_paths = [
        prim.GetPath() for prim in stage.Traverse() if prim.GetTypeName() == "Material"
    ]
    if len(material_paths) != 2:
        raise RuntimeError(
            f"Pinned Hummingbird asset changed: expected two materials, found {material_paths}"
        )

    def snapshot():
        data = []
        for prim in stage.Traverse():
            if any(prim.GetPath().HasPrefix(path) for path in material_paths):
                continue
            props = [
                (
                    p.GetName(),
                    repr(p.Get()) if isinstance(p, Usd.Attribute) else repr(p.GetTargets()),
                )
                for p in prim.GetProperties()
                if not p.GetName().startswith("material:binding")
            ]
            data.append((str(prim.GetPath()), prim.GetTypeName(), props))
        return hashlib.sha256(repr(data).encode()).hexdigest()

    before = snapshot()
    removed_bindings = []
    for prim in stage.Traverse():
        for prop in prim.GetProperties():
            if prop.GetName().startswith("material:binding"):
                removed_bindings.append(str(prop.GetPath()))
                prim.RemoveProperty(prop.GetName())
    for path in material_paths:
        stage.RemovePrim(path)
    after = snapshot()
    if before != after:
        raise RuntimeError("Visual localization changed nonmaterial model properties")
    layer.Export(str(destination))
    return {
        "adaptation": "remove external materials and bindings; retain untextured geometry",
        "upstream_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "localized_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "nonmaterial_properties_before_sha256": before,
        "nonmaterial_properties_after_sha256": after,
        "removed_materials": [str(p) for p in material_paths],
        "removed_bindings": removed_bindings,
    }
