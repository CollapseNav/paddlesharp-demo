using OpenCvSharp;
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models;
using SkiaSharp;

var appPath = AppContext.BaseDirectory;
var detPath = Path.Combine(appPath, "det");
var recPath = Path.Combine(appPath, "rec");
var clsPath = Path.Combine(appPath, "cls");

FullOcrModel model = new FullOcrModel(
    DetectionModel.FromDirectory(detPath, ModelVersion.V5),
    ClassificationModel.FromDirectory(clsPath),
    RecognizationModel.FromDirectoryV5(recPath));

using (PaddleOcrAll all = new PaddleOcrAll(model)
{
    Enable180Classification = true,
    AllowRotateDetection = true,
})
{
    string imagePath = "demo.png";
    using (Mat src = Cv2.ImRead(imagePath))
    {
        PaddleOcrResult result = all.Run(src);
        foreach (PaddleOcrResultRegion region in result.Regions)
        {
            OpenCvSharp.Point[] pts = region.Rect.Points().Select(p => new OpenCvSharp.Point(p.X, p.Y)).ToArray();
            Cv2.Polylines(src, new[] { pts }, isClosed: true, color: Scalar.Red, thickness: 2);
            if (!string.IsNullOrEmpty(region.Text))
                DrawChineseTextSafe(src, region.Text, pts[0], 20, Scalar.Blue);
            Console.WriteLine(region.Text);
        }
        using (new Window("OCR Debug", src))
        {
            Cv2.WaitKey();
        }
    }
}

static void DrawChineseTextSafe(Mat mat, string text, Point pos, int fontSize, Scalar color)
{
    using Mat bgraMat = new Mat();
    bool is3Channel = mat.Channels() == 3;

    if (is3Channel)
    {
        Cv2.CvtColor(mat, bgraMat, ColorConversionCodes.BGR2BGRA);
    }
    else
    {
        mat.CopyTo(bgraMat);
    }
    var info = new SKImageInfo(bgraMat.Width, bgraMat.Height, SKColorType.Bgra8888, SKAlphaType.Premul);

    using (var bitmap = new SKBitmap())
    {
        bitmap.InstallPixels(info, bgraMat.Data);


        using var canvas = new SKCanvas(bitmap);
        using var paint = new SKPaint()
        {
            Color = new SKColor((byte)color.Val2, (byte)color.Val1, (byte)color.Val0),
            IsAntialias = true,
        };
        using var font = new SKFont()
        {
            Size = fontSize,
            Typeface = SKTypeface.FromFamilyName("Microsoft YaHei"),
        };
        canvas.DrawText(text, pos.X, pos.Y, SKTextAlign.Left, font, paint);
    }
    if (is3Channel)
    {
        Cv2.CvtColor(bgraMat, mat, ColorConversionCodes.BGRA2BGR);
    }
    else
    {
        bgraMat.CopyTo(mat);
    }
}