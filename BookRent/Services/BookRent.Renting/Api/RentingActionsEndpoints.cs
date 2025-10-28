namespace BookRent.Renting.Api;

internal static class RentingActionsEndpoints
{
    internal static IResult ReturnBooks(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
    
    internal static IResult RentBooks(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
}