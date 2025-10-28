namespace BookRent.Renting.Api;

internal static class RentingHistoryEndpoints
{
    internal static IResult GetRentedBooks(HttpRequest request)
    {
        return TypedResults.Ok("success");
    }
    internal static IResult GetRentHistory(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
}